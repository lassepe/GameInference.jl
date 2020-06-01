include("$(@__DIR__)/../experiments/main.jl")

"Commands to run first to compile the relevant code."
function precompile_nplayer_game(np)
    (x0, solver, d) = setup_problem(np)
    (seed, game_instance) = rand(MersenneTwister(1), d)
    _, op, γ = solve(seed, game_instance, solver, x0)
    return nothing
end

# Precompile game solver
let
    foreach((2,3,5)) do np
        @info "Precompiling for $np players."
        precompile_nplayer_game(np)
    end
end

# precoompile plotting
@info "Precompiling plotting utils"
let
    problem_data = setup_problem(2)
    (seed, game_instance) = rand(MersenneTwister(1), problem_data.d_eval)
    _, op, γ = solve(seed, game_instance, problem_data.solver, problem_data.x0)
    return plot_traj(op, game_instance, GameInference.nominal_player_colors, nothing,
                     (size=(300,300), xlims=(-3.5, 3.5), ylims=(-3.5, 3.5)); kp=50)

end
