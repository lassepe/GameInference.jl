using DocStringExtensions
using StaticArrays
using Random
using ProgressMeter
using StatsPlots, DataFrames, DataFramesMeta, CSV
using ColorSchemes
using Setfield
using Statistics
using Parameters
using Clustering: kmedoids, kmeans, dbscan
using Distances: PreMetric, SqEuclidean, sqeuclidean, pairwise
using LinearAlgebra
using NamedTupleTools

using POMDPs: POMDPs

using GameInference:
    GameInference, AbstractGameDistribution, AbstractSeedDistribution,
    SeedDistributionUnicycle4D, DeterministicGameDistribution,
    ProximityCostDistribution, GoalCostDistribution, GSSDistribution,
    run_simulation, Controller, MAPController, FixedSeedController, nominal_game,
    isoptimistic, ispessimistic, isnominal, issingulargame, support, game_info

using iLQGames:
    GeneralGame, ProximityCost, Unicycle4D, NPlayerUnicycleCost,
    generate_nplayer_navigation_game, n_states, n_controls, xyindex, xindex,
    SystemTrajectory, iLQSolver, uindex, transform_to_feedbacklin, cost, plot_traj,
    plot_traj!, horizon, dynamics, player_costs, proximitycost, solve, n_players,
    ravoid, trajectory!, samplingtime, initialtime, time_disc2cont, time_cont2disc,
    @S

using Plots
using Latexify

include("plot_utils.jl")
include("montecarlo_study.jl")
include("experiment_interface.jl")
include("prediction_experiment.jl")
include("planning_experiment.jl")


"------------------------------ constants and config ------------------------------"


# TODO: it would be nicer to match this based on type dispatch but for now this is
# the qickest
function colorcode(approach::String)
    if approach in ["Inference", "InferEQ", "InferEQ+", "InferEQ 150 Particles", "InferEQ 300 Particles"]
        return :dodgerblue4
    elseif approach == "InferEQ-"
        return :cadetblue
    elseif approach in ["Baseline", "Baseline+"]
        return :dimgray
    elseif approach == "Baseline-"
        return :darkgray
    elseif approach == "InferEQOB"
        return colorant"#8b102f" #colorant"#8b104e" #
    end
    @assert false "No colorcode for approach: $approach"
end

name(c::Controller) = name(typeof(c))
name(c::Controller, problem_data::NamedTuple) = name(c, problem_data.d_ego.game_distribution)

name(::Type{<:MAPController}) = "Inference"
name(::MAPController, dgame_ego::DeterministicGameDistribution) = "InferEQ"
name(::MAPController, dgame_ego::GoalCostDistribution) = issingulargame(dgame_ego) ? "InferEQ" : "InferEQOB"
function name(::MAPController, dgame_ego::ProximityCostDistribution)
    return if isnominal(dgame_ego) || issingulargame(dgame_ego)
        "InferEQ"
    elseif isoptimistic(dgame_ego)
        "InferEQ+"
    elseif ispessimistic(dgame_ego)
        "InferEQ-"
    else
        "InferEQOB"
    end
end

name(::Type{<:FixedSeedController}) = "Baseline"
name(::FixedSeedController, dgame_ego::DeterministicGameDistribution) = "Baseline"
name(::FixedSeedController, dgame_ego::GoalCostDistribution) = "Baseline"
function name(::FixedSeedController, dgame_ego::ProximityCostDistribution)
    return if issingulargame(dgame_ego)
        "Baseline"
    elseif isoptimistic(dgame_ego)
        "Baseline+"
    elseif ispessimistic(dgame_ego)
        "Baseline-"
    else
        @assert false "Not implemented!"
    end
end

function name(experiment_type::ExperimentType, controller_problemdata_pairs::AbstractVector)
    nps = unique([n_players(nominal_game(problem_data.d_eval)) for (_, problem_data)
                  in controller_problemdata_pairs])
    @assert length(nps) == 1
    return "$(first(nps))-player-$(name(experiment_type))"
end

"---------------------------------- setup problem ---------------------------------"

function setup_problem(np=2,
                       game2distribution_eval=DeterministicGameDistribution,
                       game2distribution_ego=game2distribution_eval)
    "Computes a start and goal state from a given radius and angle."
    function start_goal_state(r, ϕ)
        # x = (x, y, phi, β, v)
        x0 = @SVector [-r*cos(ϕ), -r*sin(ϕ), ϕ, 0.0]
        xg = @SVector [ r*cos(ϕ),  r*sin(ϕ), ϕ, 0.0]
        return x0, xg
    end

    # generate a game
    T_horizon = 10.
    ΔT = 0.1
    Δϕ = np == 2 ? pi/2 : deg2rad(360/(isodd(np) ? np : np + 1))
    r_start_goal = 3
    # initial conditions and goal state
    x0s = []
    xgs = []
    for i in 1:np
        x0i, xgi = start_goal_state(r_start_goal, Δϕ*(i-1))
        push!(x0s, x0i)
        push!(xgs, xgi)
    end
    x0 = vcat(x0s...)
    # generate game
    g = generate_nplayer_navigation_game(Unicycle4D, NPlayerUnicycleCost, T_horizon,
                                         ΔT, xgs...;proximitycost=ProximityCost(1.5,
                                                                                50.0, np))
    solver = iLQSolver(g; state_regularization=5.0, control_regularization=5.0)

    seed_distribution = SeedDistributionUnicycle4D(g=g)
    d_eval = GSSDistribution(seed_distribution, game2distribution_eval(g))
    d_ego = GSSDistribution(seed_distribution, game2distribution_ego(g))

    return @namedtuple(x0, solver, d_eval, d_ego)
end

"-------------------------- data generation and plotting --------------------------"

function plot_montecarlo_study(problem_data=setup_problem(); n_particles=100,
                               show_initial_strategies=true, seed_id=1)

    objective_plts = []
    overview_plts = []
    gs = support(problem_data.d_eval.game_distribution)

    for (game_id, game_instance) in enumerate(gs)
        @info game_info(problem_data.d_eval.game_distribution, game_instance)

        d = GSSDistribution(problem_data.d_eval.seed_distribution,
                            DeterministicGameDistribution(game_instance))

        ops, ops_init, game_instances = solve_distribution(d,
                                                           problem_data.solver,
                                                           problem_data.x0, n_particles,
                                                           MersenneTwister(seed_id))
        ps = plotop_clustered(ops, game_instances, f_sampled_identity;
                              ops_init=(show_initial_strategies ?  ops_init :
                                        nothing),
                              cid_titleprefix=(length(gs) > 1 ? "$game_id." : ""))

        push!(objective_plts, plot_rowise(ps))
        push!(overview_plts, title!(deepcopy(first(ps)), "$(id2figchar(game_id)) Cluster $game_id.1"))
    end


    plts_with_anno = [(p, "$i") for (i, p) in enumerate(objective_plts)]
    # if there is more than one objective, also create an overview
    if length(gs) > 1
        push!(plts_with_anno, (plot_rowise(overview_plts; n_cols=2), "overview"))
    end

    return plts_with_anno
end

function plot_example_trajectory(problem_data=setup_problem(3); seed_id=1)
    (seed, game_instance) = rand(MersenneTwister(seed_id), problem_data.d_eval)
    _, op, γ = solve(seed, game_instance, problem_data.solver, problem_data.x0)
    return plot_traj(op, game_instance, GameInference.nominal_player_colors, nothing,
                     (size=(300,300), xlims=(-3.5, 3.5), ylims=(-3.5, 3.5)); kp=50)
end

function plot_example_planning(ego_basetype=FixedSeedController,
                               problem_data=setup_problem(3); seed_id=3,
                               save_fig_path=joinpath(datadir_uneq,
                                                      "planning_misaligned_demo",
                                                      name(ego_basetype)),
                               kwargs...)
    run_experiment(PlanningExperiment(), ego_basetype, problem_data, MersenneTwister(seed_id);
                   visualize=true, verbose=true, save_fig_path=save_fig_path,
                   kwargs...)
end

"------------------------------ Uncertain Equilibria ------------------------------"

datadir_uneq = joinpath(@__DIR__, "../results/uncertain-equilibria")
approaches_uneq = [MAPController, FixedSeedController]
experiment_setups_uneq = [
                          setup_experiment(PredictionExperiment(), setup_problem(3),
                                           approaches_uneq),
                          setup_experiment(PlanningExperiment(), setup_problem(3),
                                           approaches_uneq),
                          setup_experiment(PlanningExperiment(), setup_problem(5),
                                           approaches_uneq;
                                           run_kwargs=(n_particles=150,)),
                         ]

function generate_plots_uneq(datadir=datadir_uneq)

    @info "monte carlo study"
    saveplts(plot_montecarlo_study(setup_problem(2)),
            datadir, "2-player-montecarlo")
    saveplts(plot_montecarlo_study(setup_problem(3)),
            datadir, "3-player-montecarlo")

    @info "quantitative experiments"
    foreach(experiment_setups_uneq) do ex
        create_and_save_viz(ex.experiment_type, ex.controller_problemdata_pairs,
                            datadir; ex.viz_kwargs...)
    end

    @info "qualitative experiments and examples"
    plot_example_planning(FixedSeedController, setup_problem(3))
    plot_example_planning(MAPController, setup_problem(3))
    # example trajectory
    p_example = plot([plot_example_trajectory(setup_problem(5); seed_id=i) for i in
                      [1,2,5,6]]...; layout=(2,2), size=(600,600))
    savefig(p_example, joinpath(datadir, "5-player-example.pdf"))
end

function run_uneq()
    run_and_save_experiments(experiment_setups_uneq , datadir_uneq)
    generate_plots_uneq()
end

function compare_5player_nparticles(datadir=datadir_uneq)
    df_less = CSV.read("$datadir/5-player-planning.csv")
    df_more = CSV.read("$datadir/5-player-planning-more-particles.csv")
    return compare_infereq(df_less, df_more)
end

function compare_infereq(df_less, df_more)
    select_and_rename(df, new_name) = @linq df |> where(:ego .== "InferEQ") |> transform(ego = [new_name for _ in :ego])

    df = vcat(select_and_rename(df_less, "InferEQ 150 Particles"),
              select_and_rename(df_more, "InferEQ 300 Particles"))

    visualize(PlanningExperiment(), df)

end

"------------------------------ Uncertain Proximity -------------------------------"

datadir_unprox = joinpath(@__DIR__, "../results/uncertain-objectives/unprox")
# the robot is aware of both types of objectives
d_unprox_true = g->ProximityCostDistribution(g, [50., 10.])
problem_data_unprox_aware = setup_problem(3, d_unprox_true)
# the robot always assumes that the other player is aware of it
problem_data_unprox_optimistic = setup_problem(3, d_unprox_true,
                                             g->ProximityCostDistribution(g, [50.]))
#problem_data_unprox_random = setup_problem(3, d_unprox_true,
#                                           g->ProximityCostDistribution(g,
#                                                                        [100., 50.,
#                                                                        10.];
#                                                                        singular=true))
#
# the robot always assumes that the other player is not aware of it
problem_data_unprox_pessimistic = setup_problem(3, d_unprox_true,
                                              g->ProximityCostDistribution(g, [10.]))

# approaches
approaches_unprox = [
                     (MAPController, problem_data_unprox_aware), # infer cost + eq
                     (MAPController, problem_data_unprox_pessimistic), # infer eq + optimistc cost
                     (MAPController, problem_data_unprox_optimistic), # infer eq + optimistc cost
                     (FixedSeedController, problem_data_unprox_pessimistic), # baseline eq + optimistic cost
                     (FixedSeedController, problem_data_unprox_optimistic), # infer eq + optimistc cost
                    ]

experiment_setups_unprox = [
                            setup_experiment(PredictionExperiment(), approaches_unprox;
                                             run_kwargs=(n_particles=150,)),
                            setup_experiment(PlanningExperiment(), approaches_unprox;
                                             run_kwargs=(n_particles=150,),
                                             viz_kwargs=(size=(800,300),)),
                         ]

function generate_plots_unprox(datadir=datadir_unprox)
    @info "monte carlo study"
    saveplts(plot_montecarlo_study(problem_data_unprox_aware; n_particles=300),
             datadir, "3-player-montecarlo")

    @info "quantitative experiments"
    for (experiment_type, problem_data, _, kwargs) in experiment_setups_unprox
        create_and_save_viz(experiment_type, problem_data, datadir; kwargs...)
    end
end

function run_unprox()
    run_and_save_experiments(experiment_setups_unprox, datadir_unprox)
    generate_plots_unprox()
end

"--------------------------------- Uncertain Goals --------------------------------"

datadir_ungoal = joinpath(@__DIR__, "../results/uncertain-objectives/ungoal")
d_ungoal_true = GoalCostDistribution
problem_data_ungoal_aware = setup_problem(3, d_ungoal_true)
problem_data_ungoal_random = setup_problem(3, d_ungoal_true,
                                           g->GoalCostDistribution(g; singular=true))
approaches_ungoal = [
                     (MAPController, problem_data_ungoal_aware),
                     (MAPController, problem_data_ungoal_random),
                     (FixedSeedController, problem_data_ungoal_aware)
                    ]
experiment_setups_ungoal = [
                            setup_experiment(PredictionExperiment(), approaches_ungoal;
                                             run_kwargs=(n_particles=150,)),
                            setup_experiment(PlanningExperiment(), approaches_ungoal;
                                             run_kwargs=(n_particles=150,)),
                           ]

function generate_plots_ungoal(datadir=datadir_ungoal)
    saveplts(plot_montecarlo_study(problem_data_ungoal_aware; n_particles=300),
             datadir, "3-player-montecarlo")

    @info "quantitative experiments"
    for (experiment_type, problem_data, _, kwargs) in experiment_setups_ungoal
        create_and_save_viz(experiment_type, problem_data, datadir; kwargs...)
    end
end

function run_ungoal()
    run_and_save_experiments(experiment_setups_ungoal, datadir_ungoal)
    generate_plots_ungoal()
end
