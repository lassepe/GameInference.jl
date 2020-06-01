
function plotop_clustered(ops, game_instances, f_map; cluster_metric=SqEuclidean(),
                          ops_init = nothing, cid_titleprefix="")

    nominal_game = first(game_instances)

    #k_clusters = Int(2^(n_players(g)*(n_players(g) - 1)/2))
    f_mat = f_map(ops, nominal_game)
    # extract feature coordinates
    f1 = f_mat[1, :]
    f2 = f_mat[2, :]

    # Info
    pcosts = pairwise(cluster_metric, f_mat, dims=2)
    clusters = dbscan(pcosts, 10.0, 2)
    @show k_clusters = length(clusters.counts)

    # base pots for clustered trajectories
    p_trajclusters = [plot() for i in 1:k_clusters]
    p_extraplots = []
    median_costs = resize!([], k_clusters)

    # sort element ids by cluster
    ids_by_cluster = [[] for i in 1:k_clusters]
    for (i, ci) in enumerate(clusters.assignments)
        if ci == 0
            println("outlier")
            continue
        end
        push!(ids_by_cluster[ci], i)
    end

    # compute costs:
    for (ci, ids) in enumerate(ids_by_cluster)
        ops_ci = ops[ids]
        games_ci = game_instances[ids]
        costs_ci = map((op,gi)->cost(gi, op), ops_ci, games_ci)
        median_costs[ci] = [median(pcs[i] for pcs in costs_ci) for i in
                            1:n_players(nominal_game)]
    end


    # config
    sorting_criterion = [first(ci) for ci in median_costs]
    cost_colorschemes = reverse.([ColorSchemes.PuBu_9,
                                  ColorSchemes.Reds_9,
                                  ColorSchemes.Purples_9,
                                  ColorSchemes.Greens_9,
                                  ColorSchemes.Oranges_9])
    cost_colorrange = extrema(Iterators.flatten(sorting_criterion)) .+ (-500, 500)

    for (ci , ids) in enumerate(ids_by_cluster)
        ops_ci = ops[ids]
        games_ci = game_instances[ids]
        playercosts = median_costs[ci]
        player_colors = [get(cost_colorschemes[i], pci, cost_colorrange)
                         for (i, pci) in enumerate(playercosts)]

        # plot all trajectories for this cluster
        for (op,gi) in zip(ops_ci, game_instances)
            # the second point to be marked on the trajectory
            kp = (n_players(gi) == 2 ?
                  last(findpairwisemindist(op, xyindex(gi))) : 50)
            plot_traj!(p_trajclusters[ci], op, gi, GameInference.nominal_player_colors, nothing,
                       (xlims=(-3.5, 3.5), ylims=(-3.5, 3.5)); k=1, kp=kp)
        end
    end

    # sort plots by cost for player 1
    p_sorted_trajclusters = [title!(p, "$(id2figchar(i)) Cluster $cid_titleprefix$i")
                             for (i, (_, p)) in enumerate(sort(collect(zip(sorting_criterion,
                                                              p_trajclusters))))]

    if !isnothing(ops_init)
        p_ops_init = plot(;title="($('a'+k_clusters)) Initial Strategies")
        for oi in ops_init
            plot_traj!(p_ops_init, oi, nominal_game,
                       GameInference.nominal_player_colors, nothing,
                       (seriesalpha=0.1, xlims=(-3.5, 3.5), ylims=(-3.5, 3.5)))
        end
        push!(p_extraplots, p_ops_init)
    end

    return vcat(p_sorted_trajclusters, p_extraplots)
end



"---------------------- Solve For Distribution of Strategies ----------------------"
function solve_distribution(d::GSSDistribution, solver::iLQSolver, x0,
                            n_samples::Int, rng::AbstractRNG,
                            max_iter::Int=2*n_samples)

    ops = []
    ops_init = []
    game_instances = []
    iter = 0

    while length(ops) < n_samples && iter < max_iter
        iter += 1
        seed, game_instance = rand(rng, d)

        _, op, _ = solve(seed, game_instance, solver, x0)
        if all(mindist(op, xyindex(game_instance)) .> ravoid(game_instance) * 0.5)
            op_init = zero(op)
            trajectory!(op_init, dynamics(game_instance), seed, zero(op), x0)
            push!(ops, op)
            push!(ops_init, op_init)
            push!(game_instances, game_instance)
        end
    end

    if length(ops) < n_samples
        @warn "Sampled only $(length(ops))/$n_samples"
    else
        @info "Sampled $(length(ops)) after $iter iterations."
    end

    return ops, ops_init, game_instances
end

"------------------------------- Trajectory Features ------------------------------"

"Two player comparison of minimum distance. Returns tuple of distance and time
index."
function findpairwisemindist(op::SystemTrajectory, xyis::NTuple{2})
    return map(op.x) do x
        return norm(x[xyis[1]] - x[xyis[2]])
    end |> findmin
end
function findmindist(op::SystemTrajectory, xyis::NTuple)
    return [i == j ? (Inf, 0) : findpairwisemindist(op, (xyi, xyj))
            for (i, xyi) in enumerate(xyis), (j, xyj) in enumerate(xyis)]
end
mindist(op::SystemTrajectory, xyis::NTuple) = first.(findmindist(op, xyis))

function relvec(op::SystemTrajectory, xyis, k::Int)
    @assert length(xyis) == 2
    return op.x[k][xyis[2]] - op.x[k][xyis[1]]
end

"-------------------------------- ExtractFeatures ---------------------------------"

# Other features to try:
# - absolute position of both players at closest point
# - absolute position of ego at closest point
# - cost

"Maps each trajectory to the relative vector at the point `k` where both players are
closest"
function f_msevec(ops, g)
    xyids = xyindex(g)
    hcat(map(ops)
         do op
             (d_min, k_min) = findpairwisemindist(op, xyids)
             return relvec(op, xyids, k_min)
         end...)
end

"Maps each trajectory to subsampled and flattend version (high dimensional feature)"
function f_sampled_identity(ops, g, inter_sample_dist::Int=1)
    xyids = xyindex(g)
    joint_xyids = vcat(xyids...)
    op_features = map(ops) do op
        subsampled_op = [x for (i, x) in enumerate(op.x)
                         if iszero(i % inter_sample_dist)]
        return collect(Iterators.flatten(subsampled_op))
    end

    return hcat(op_features...)
end

function f_cost(ops, g)
    op_features = map(ops) do op
        return cost(g, op)
    end

    return hcat(op_features...)
end

