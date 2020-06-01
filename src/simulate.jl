"--------------------------------- Initial Belief ---------------------------------"

function initial_belief(rng::AbstractRNG, d, x0, n_particles)

    # a system trajectory that consists of all x0 states and zero input
    x0_op = zero(SystemTrajectory, nominal_game(d))
    fill!(x0_op.x, x0)
    s0, g0 = rand(rng, d)

    gss_vec = map(1:n_particles) do _
        seed, game_instance = rand(rng, d)
        return GSS(copy(x0_op),
                   issingularseed(d) ? s0 : seed,
                   issingulargame(d) ? g0 : game_instance)
    end |> SizedVector{n_particles}

    gss_cache = GSSCache(gss_vec)

    return ParticleCollection([CachingGSS(gss_cache, i) for i in 1:length(gss_cache)])
end

"----------------------------- Controller Abstraction -----------------------------"

# TODO: we should cast the whole problem in the POMDPs.jl if possible
abstract type Controller{uid} end
function controlled_inputs end
function control end
function initial_planning_state end
function prediction end
function update_planning_state end
controlled_inputs(c::Controller{uid}) where {uid} = uid

struct MAPController{uid} <: Controller{uid} end
initial_planning_state(c::MAPController, b, rng::AbstractRNG) = b
function control(c::MAPController, m::GameSolutionModel, b)
    # find the most likely state
    gss = sorted_mode(b)
    # solve the game for this state
    solve_with_fallback!(gss, m)
    return first(operating_point(gss).u)
end
prediction(c::MAPController, b) = operating_point(sorted_mode(b))
function update_planning_state(c::MAPController, b_planning, b, x, model,
                               rng::AbstractRNG)
    # TODO: for now, this assmes that the belief is updated outside
    @assert all(external_updated(p) && internal_updated(p) for p in particles(b))
    return b
end

struct FixedSeedController{uid} <: Controller{uid} end
function initial_planning_state(c::FixedSeedController, b, rng::AbstractRNG)
    return statelabel(rand(rng, b))
end
function control(c::FixedSeedController, m::GameSolutionModel, gss)
    solve_with_fallback!(gss, m)
    return first(operating_point(gss).u)
end
prediction(c::FixedSeedController, gss) = operating_point(gss)
function update_planning_state(c::FixedSeedController, gss, b, x, model,
                               rng::AbstractRNG)
    gss = POMDPs.gen(model, gss, 0., rng).sp
    return update_external!(gss, x)
end

"-------------------------------- Simulation Method -------------------------------"

function run_simulation(x0, solver, d_eval, d_ego, eval_controller, ego_controller,
                        rng; n_particles=50, infer_only=false, visualize=true,
                        verbose=true, save_fig_path=nothing)

    b_eval = initial_belief(rng, d_eval, x0, n_particles)
    b_ego = initial_belief(rng, d_ego, x0, n_particles)

    # sample the sevaluation player
    eval_planning_state = initial_planning_state(eval_controller, b_eval, rng)
    ego_planning_state = initial_planning_state(ego_controller, b_ego, rng)
    true_game = game_instance(eval_planning_state)

    zero_op = zero(SystemTrajectory, true_game)
    nu = n_controls(true_game)
    xids = xindex(true_game)
    xyids = xyindex(true_game)

    # Given the dynamics and observation model, we can construct the particle filter
    # model.
    snm = (x, rng) -> x
    om = infer_only ? identity : x -> x[(last(first(xids))+1):end]
    S = Random.gentype(b_ego); A = Float64; O = typeof(x0);
    model = GameSolutionModel{S, A, O}(solver, om)

    # construct the filter.
    pf = SIRParticleFilter(model, n_particles, rng)

    # state sequence to record the closed loop trjectories
    # read this from the game
    n_clsteps = horizon(true_game)+1
    x_cl = SizedVector{n_clsteps, eltype(zero_op.x)}(undef)
    u_cl = SizedVector{n_clsteps, eltype(zero_op.u)}(undef)
    ΔT_cl = samplingtime(true_game)
    t0_cl = 0.0
    traj_cl = SystemTrajectory{ΔT_cl}(x_cl, u_cl, t0_cl)

    # current state
    xₖ = x0

    # buffers to record the predictions
    ego_predictions = SizedVector{n_clsteps, typeof(zero_op)}(undef)

    # abusing GR as a gui for real time plotting
    Δt_frame= 0.1
    t_last = time()
    for k in 1:n_clsteps
        # extract the control input that each player will take
        u_ego = control(ego_controller, model, ego_planning_state)
        u_eval = control(eval_controller, model, eval_planning_state)
        utmp = @MVector zeros(nu)
        if infer_only
            uₖ = u_eval
        else
            utmp[controlled_inputs(ego_controller)] = u_ego[controlled_inputs(ego_controller)]
            utmp[controlled_inputs(eval_controller)] = u_eval[controlled_inputs(eval_controller)]
            uₖ = SVector(utmp)
        end
        tₖ = time_disc2cont(traj_cl, k)
        traj_cl.x[k] = xₖ
        traj_cl.u[k] = uₖ

        # simulate the next state
        xₖ = next_x(dynamics(true_game), xₖ, uₖ, tₖ)
        o = om(xₖ)

        # belief update if anyone is using it
        uses_belief = any(c isa MAPController for c in (ego_controller, eval_controller))
        if uses_belief
            b_ego = update(pf, b_ego, 0.0, o)
            merge_similar!(b_ego, 0.1)
            update_external!(b_ego, xₖ)
        end

        # update eval player
        eval_planning_state = update_planning_state(eval_controller,
                                                    eval_planning_state, b_ego, xₖ,
                                                    model, rng)
        eval_prediction = prediction(eval_controller, eval_planning_state)
        # update ego player
        ego_planning_state = update_planning_state(ego_controller,
                                                   ego_planning_state, b_ego, xₖ,
                                                   model, rng)
        ego_prediction = prediction(ego_controller, ego_planning_state)
        reset_updated!(b_ego)

        # record predictions
        ego_predictions[k] = copy(ego_prediction)

        if visualize
            true_plan_colors = [colorant"palevioletred2" for i in 1:5]
            belief_plan_colors = [colorant"dodgerblue4" for i in 1:5]
            ego_plan_colors = [colorant"black" for i in 1:5]

            plt_traj = plot_traj(eval_prediction, true_game, true_plan_colors,
                                 nothing, (line=(:dash, 3),); k=0, kp=0)
            # plot the particle belief in blues
            if uses_belief
                for p in particles(b_ego)
                    plot_traj!(plt_traj, operating_point(p), true_game,
                               belief_plan_colors, nothing, (seriesalpha=0.05,);
                               k=0, kp=0)
                end
            end
            plot_traj!(plt_traj, ego_prediction, true_game, ego_plan_colors,
                      nothing, (line=2, seriesalpha=0.5); k=0, kp=0)
            scatter_positions!(plt_traj, xₖ, true_game, nominal_player_colors,
                               (:circle, 5))

            save_fig_only = !isnothing(save_fig_path)

            if k % 10 == 1 || !save_fig_only
                if uses_belief
                    # plot belief hisotgram
                    plt_hist = histogram([p.id for p in particles(b_ego)],
                                          bins=1:n_particles+1,
                                          legend=false,
                                          normed=true,
                                          title="Particle Belief",
                                          xlabel="particle index",
                                          ylabel="weight",
                                          seriescolor=colorant"dodgerblue4")
                    p = plot(plt_traj, plt_hist;
                             layout=grid(2, 1, heights=[0.75, 0.25]),
                             size=(400, 500))
                else
                    p = plot(plt_traj, size=(400,400))
                end
                if save_fig_only
                    savefig(p, "$(save_fig_path)/sim-$(lpad(k, 3, "0")).pdf")
                else
                    display(p)
                end
            end

            sleep(max(0., Δt_frame - (time() - t_last)))
            t_last = time()
        end
        if verbose
            print(".")
        end
    end

    return traj_cl, ego_predictions, true_game
end
