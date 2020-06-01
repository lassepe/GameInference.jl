struct PredictionExperiment <: ExperimentType end
name(::PredictionExperiment) = "prediction"
result_df_template(::PredictionExperiment) = DataFrame(run_id=Int[],
                                                       prediction_error=Float64[],
                                                       closedloop_start=Int[],
                                                       closedloop_step=Int[],
                                                       prediction_step=Int[],
                                                       ego=String[])

"Run a single experiment of a predictor."
function run_experiment(::PredictionExperiment, ego_basetype=FixedSeedController,
                        problem_data=setup_problem(3), rng=MersenneTwister(1),
                        run_id=-1; kwargs...)
    # here, the ego only observes. Thus the input indeces are set to "nothing"...
    ego_controller = ego_basetype{nothing}()
    # ...while the "eval player" controls all the inputs
    eval_controller = FixedSeedController{@S(1:n_controls(nominal_game(problem_data.d_eval)))}()
    traj_cl, predictions = run_simulation(problem_data..., eval_controller,
                                          ego_controller, rng; infer_only=true,
    kwargs...)

    prederrors = map(predictions[1:end-1]) do pred
        prediction_error(pred, traj_cl, xyindex(nominal_game(problem_data.d_eval)))
    end

    result = []

    for (kcl_start, err_traj) in enumerate(prederrors),
        (kpred, err) in enumerate(err_traj)

        push!(result, (run_id=run_id, prediction_error=err,
                       closedloop_start=(kcl_start-1),
                       closedloop_step=(kcl_start+kpred-1),
                       prediction_step=kpred,
                       ego=name(ego_controller, problem_data)))
    end

    return result
end

function visualize(experiment_type::PredictionExperiment, df::DataFrame)
    prederrorstats(df::DataFrame, time_group::Symbol) = @linq df |>
        by([time_group, :ego],
           mean=mean(:prediction_error),
           sem=(std(:prediction_error)/sqrt(length(:prediction_error))))

    plt_setups = [(anno="trajerror",
                   stats=prederrorstats(df, :closedloop_start),
                   kwargs=(xlabel="time [s]",
                           ylabel="mean squared prediction error [m²]")),
                  (anno="trajerror-short",
                   stats=(@linq df |> where(:prediction_step .<= 20)) |>
                       d->prederrorstats(d, :closedloop_start),
                   kwargs=(xlabel="time [s]",
                           ylabel="mean squared prediction error [m²]")),
                  (anno="prederror",
                   stats= (@linq df |> where(:prediction_step .<= Inf)) |>
                       d->prederrorstats(d, :prediction_step),
                   kwargs=(xlabel="look ahead [s]",
                           ylabel="mean squared prediction error [m²]"))]

    return map(plt_setups) do ps
        plt = plot(;size=(600,300), ps.kwargs...)
        for ego in unique(ps.stats[:, :ego])
            stats_df_pred = @linq ps.stats |> where(:ego .== ego)
            plot!(plt, stats_df_pred[:, 1].*0.1, stats_df_pred.mean;
                  ribbon=stats_df_pred.sem, label=ego, seriescolor=colorcode(ego))
        end

        return plt, ps.anno
    end
end

"""
    $(TYPEDSIGNATURES)

Computes the prediction error of the predicted trajectory `ptraj` with respect
to the true closed-loop trajectory cltraj.
"""
function prediction_error(ptraj::SystemTrajectory, cltraj::SystemTrajectory, xyids)
    @assert samplingtime(ptraj) == samplingtime(cltraj)
    # initial time
    ti_ptraj = initialtime(ptraj)
    ti_cltraj = initialtime(cltraj)
    # final time
    tf_ptraj = time_disc2cont(ptraj, horizon(ptraj))
    tf_cltraj = time_disc2cont(cltraj, horizon(cltraj))

    @assert(ti_ptraj >= ti_cltraj, "Prediction must start after ground truth.")
    @assert(ti_ptraj <= tf_cltraj, "Trajectories must be overlapping in time.")

    # 1. find the segement of the `ptraj` for wich we have `cltraj` ground truth
    tf_overlap = clamp(tf_ptraj, ti_cltraj, tf_cltraj)
    # extract start and end discrete time index
    ki_ptraj = 1
    kf_ptraj = time_cont2disc(ptraj, tf_overlap)
    ki_cltraj = time_cont2disc(cltraj, ti_ptraj)
    kf_cltraj = time_cont2disc(cltraj, tf_overlap)

    @assert kf_ptraj <= horizon(ptraj)
    @assert ki_cltraj >= 1
    @assert kf_cltraj <= horizon(cltraj)
    @assert (kf_ptraj - ki_ptraj) == (kf_cltraj - ki_cltraj)

    # 2. Cumpute the distance between these segments
    xyis = vcat(xyids...)
    ppositions = (x[xyis] for x in ptraj.x[ki_ptraj:kf_ptraj])
    clpositions = (x[xyis] for x in cltraj.x[ki_cltraj:kf_cltraj])
    return sqeuclidean.(ppositions, clpositions)
end
