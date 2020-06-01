module GameInference
    using Parameters
    using StaticArrays
    using LinearAlgebra
    using Statistics
    using Random
    using DocStringExtensions
    using Plots, ColorSchemes

    using ParticleFilters:
        AbstractParticleBelief,
        ParticleCollection,
        WeightedParticleBelief,
        LowVarianceResampler,
        SIRParticleFilter,
        n_particles,
        particle,
        particles,
        mode,
        update
    using POMDPs: POMDP
    using Distributions: MvNormal, pdf
    using Setfield

    # imported for extension
    import POMDPs: POMDPs, gen
    import POMDPModelTools: POMDPModelTools, obs_weight, SparseCat
    import ParticleFilters: ParticleFilters, resample
    import Base: Base, length, circshift!
    import iLQGames: iLQGames, horizon, n_states, n_controls, n_players

    using iLQGames:
        are_close,
        AbstractGame,
        GeneralGame,
        trajectory!,
        SystemTrajectory,
        iLQSolver,
        solve,
        solve!,
        n_states,
        n_controls,
        dynamics,
        zero!,
        next_x,
        AffineStrategy,
        plot_traj!,
        plot_traj,
        xyindex,
        xindex,
        uindex,
        statetype,
        strategytype,
        samplingtime,
        initialtime,
        horizon,
        time_disc2cont,
        time_cont2disc,
        cost,
        ravoid,
        @S,
        scatter_positions!,
        NPlayerUnicycleCost,
        player_costs

    const nominal_player_colors = [:steelblue, :red3, :mediumpurple, :seagreen,
                                   :darkorange]

    include("distributions.jl")
    include("inference.jl")
    include("simulate.jl")
end # module
