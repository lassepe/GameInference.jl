"Returns the name of the experiment (for file naming etc.)"
function name end

abstract type ExperimentType end
"Returns an empty DataFrame with the column names for this experiment type."
function result_df_template end
"Returns a collection of dataframe-rows / named-tuples for a single experiment run."
function run_experiment end
"Creates a plot to visualize the results of the experiment"
function visualize_results end


"--------------------------------- implementations --------------------------------"

function setup_experiment(experiment_type::ExperimentType,
                          controller_problemdata_pairs; run_kwargs=NamedTuple(),
                          viz_kwargs=NamedTuple())
    return @namedtuple(experiment_type, controller_problemdata_pairs, run_kwargs,
                       viz_kwargs)
end

function setup_experiment(experiment_type::ExperimentType, problem_data,
                          controllers; kwargs...)
    return setup_experiment(experiment_type,
                            [@namedtuple(controller, problem_data) for controller in
                             controllers]; kwargs...)
end

"Run a bunch of experiments in a multi-threaded fashion."
function run_experiments(experiment_type::ExperimentType, ego_basetype=FixedSeedController,
                         problem_data=setup_problem(3); n_samples=200, kwargs...)
    result_df = result_df_template(experiment_type)

    thread_problem_data = [deepcopy(problem_data) for i in 1:Threads.nthreads()]
    progmeter = Progress(n_samples)
    result_lock = ReentrantLock()

    Threads.@threads for i in 1:n_samples
        result = run_experiment(experiment_type, ego_basetype,
                                thread_problem_data[Threads.threadid()],
                                MersenneTwister(i), i; visualize=false,
                                verbose=false, kwargs...)
        lock(result_lock)
        append!(result_df, result)
        next!(progmeter)
        unlock(result_lock)
    end

    return result_df
end

function run_and_save_experiments(experiment_setups, datadir)
    foreach(experiment_setups) do ex
        run_and_save_experiment(ex.experiment_type, ex.controller_problemdata_pairs,
                                datadir; ex.run_kwargs...)
    end
end

function run_and_save_experiment(experiment_type, controller_problemdata_pairs,
                                 datadir; kwargs...)

    experiment_name = name(experiment_type, controller_problemdata_pairs)
    @info "Running: $experiment_name"
    @info "Kwargs: $(collect(kwargs))"

    results = map(controller_problemdata_pairs) do (controller, problem_data)
        run_experiments(experiment_type, controller, problem_data; kwargs...)
    end

    df = vcat(results...)
    CSV.write(joinpath(datadir, "$experiment_name.csv"), df)
end

function create_and_save_viz(experiment_type, controller_problemdata_pairs, datadir;
                             kwargs...)
    experiment_name = name(experiment_type, controller_problemdata_pairs)
    @info "Visualizing: $experiment_name"
    @info "Kwargs: $(collect(kwargs))"

    file_basename = "$(joinpath(datadir, experiment_name))"
    df = CSV.read("$file_basename.csv")
    ps = visualize(experiment_type, df; kwargs...)
    if ps isa Plots.Plot
        savefig(ps, "$file_basename.pdf")
    else
        foreach(ps) do (p, anno)
            savefig(p, "$file_basename-$anno.pdf")
        end
    end
end

