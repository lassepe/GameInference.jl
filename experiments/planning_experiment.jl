struct PlanningExperiment <: ExperimentType end
name(::PlanningExperiment) = "planning"
result_df_template(::PlanningExperiment) = DataFrame(run_id=Int[], cost=Float64[], dmin=Float64[],
                                                     player_id=Int[], ego=String[])
"Run a single simulation of a planner."
function run_experiment(::PlanningExperiment, ego_basetype=FixedSeedController,
                        problem_data=setup_problem(3), rng=MersenneTwister(1),
                        run_id=-1; kwargs...)
    # setup controllers
    uids = uindex(nominal_game(problem_data.d_eval))
    uid_ego = first(uids)
    uid_eval = vcat(uids[2:end]...)
    ego_controller = ego_basetype{uid_ego}()
    eval_controller = FixedSeedController{uid_eval}()

    # run planner
    traj_cl, _, true_game = run_simulation(problem_data..., eval_controller,
                                           ego_controller, rng; kwargs...)

    # post processing / compsing result for dataframe
    #TODO: think about what is the right cost to use here?
    costs = cost(true_game, traj_cl)
    mds = mindist(traj_cl, xyindex(true_game))
    return [(run_id=run_id, cost=c, dmin=minimum(mds[:, i]), player_id=i,
             ego=name(ego_controller,  problem_data))
             for (i, c) in enumerate(costs)]
end

"Create a violin plot with a transparant box plot on top."
function boxviolin(df::DataFrame, xkey::Symbol, ykey::Symbol; group::Symbol,
                   range::Real=1.5, kwargs...)
    plts = []

    cmin, cmax = Inf, -Inf

    for (i, g) in enumerate(unique(df[:, group]))
        gcolor = colorcode(string(g))
        dfg = filter(df) do row
            row[group] == g
        end
        dfg_trunc = DataFrame()
        for xki in unique(dfg[:, xkey])
            dfg_xki = filter(dfg) do row
                row[xkey] == xki
            end
            q1, q3 = quantile(dfg_xki[:, ykey], (0.25, 0.75))
            iQR = q3-q1
            plow, phigh = quantile(dfg_xki[:, ykey], (0.02, 0.98))
            cpmin = min(q1-range*iQR, plow)
            cpmax = max(q3+range*iQR, phigh)

            dfg_xki_filtered = filter(dfg_xki) do row
                cpmin <= row[ykey] <= cpmax
            end
            dfg_trunc = vcat(dfg_trunc, dfg_xki_filtered)
        end
        cmin = min(cmin, minimum(dfg_trunc[:, ykey]))
        cmax = max(cmax, maximum(dfg_trunc[:, ykey]))

        # density truncated to whisker range
        p = violin(dfg_trunc[:, xkey], dfg_trunc[:, ykey]; title="$(id2figchar(i)) $(string(g))",
                   xlabel=string(xkey), ylabel=string(ykey), color=gcolor)
        boxplot!(p, dfg[:, xkey], dfg[:, ykey]; alpha=0.5, range=range,
                 outliers=false, color=gcolor)
        push!(plts, p)

    end

    cmin = round(cmin, RoundDown; digits=-2)
    cmax = round(cmax, RoundUp; digits=-2)

    return plot(plts...; layout=(1, length(plts)), link=:all, ylims=(cmin, cmax), legend=false, kwargs...)
end

function visualize(experiment_type::PlanningExperiment, df::DataFrame; kwargs...)
    np = maximum(df[:, :player_id])
    boxviolin(df, :player_id, :cost; group=:ego, size=(600,300), xlabel="player", ylabel="cost",
              xticks=([1:np;], ["P$i" for i in 1:np]), kwargs...)
end
