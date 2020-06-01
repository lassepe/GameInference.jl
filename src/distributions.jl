"-------------------------------- Seed Distribution -------------------------------"

abstract type AbstractSeedDistribution end

steer_profile(k::Int, h::Int) = cos(k/h*pi)
acc_profile(k::Int, h::Int) = -cos(k/h*pi) * 0.05

@with_kw struct SeedDistributionUnicycle4D{TG<:GeneralGame} <: AbstractSeedDistribution
    g::TG
    steer_range::Tuple{Float64, Float64} = (deg2rad(-15), deg2rad(15))
    acc_scale_noise::Float64 = 0.5
end

function Base.rand(rng::AbstractRNG, d::SeedDistributionUnicycle4D)
    @unpack g, steer_range, acc_scale_noise = d
    steer_min, steer_max = steer_range
    np = n_players(g)
    nx = n_states(g)
    nu = n_controls(g)
    h = horizon(g)

    steer_scale = (steer_max - steer_min)*[rand(rng) for i in 1:np] .+ steer_min
    acc_scale = 2*acc_scale_noise*[1+(rand(rng)-0.5) for i in 1:np] .+ 1

    return SizedVector{h}([AffineStrategy(@SMatrix(zeros(nu, nx)),
                                         SVector{nu}(vcat(([steer_profile(k, h)*steer_scale[i],
                                                            acc_profile(k, h)*acc_scale[i]]
                                                           for i in 1:np)...))) for k in
                          1:h])
end

@with_kw struct SeedDistributionUnicycle4DFlat{TG} <: AbstractSeedDistribution
    g::TG
    acc_x_range::Tuple{Float64, Float64} = (-0.1, 0.1)
    acc_y_range::Tuple{Float64, Float64} = (-0.1, 0.1)
end

function Base.rand(rng::AbstractRNG, d::SeedDistributionUnicycle4DFlat)
    @unpack g, acc_x_range, acc_y_range = d
    acc_x_min, acc_x_max = acc_x_range
    acc_y_min, acc_y_max = acc_y_range
    np = n_players(g)
    nx = n_states(g)
    nu = n_controls(g)
    h = horizon(g)

    acc_x = (acc_x_max - acc_x_min)*[rand(rng) for i in 1:np] .+ acc_x_min
    acc_y = (acc_y_max - acc_y_min)*[rand(rng) for i in 1:np] .+ acc_y_min

    return SizedVector{h}([AffineStrategy(@SMatrix(zeros(nu, nx)),
                                          SVector{nu}(vcat(([acc_x[i], acc_y[i]] for i in
                                                            1:np)...))) for k in 1:h])
end


"-------------------------------- Game Distribution -------------------------------"


"Implements rand(rng, d, g) which returns a new game instance based on the nominal
game g."
abstract type AbstractGameDistribution end
function support end
function nominal_game end
function issingulargame end
function game_info end

struct DeterministicGameDistribution{TG} <: AbstractGameDistribution
    g::TG
end
nominal_game(d::DeterministicGameDistribution) = d.g
support(d::DeterministicGameDistribution) = [d.g]
game_info(::DeterministicGameDistribution, ::GeneralGame) = ""
issingulargame(d::DeterministicGameDistribution) = true
Base.rand(rng::AbstractRNG, d::DeterministicGameDistribution) = d.g

"Special case of AbstractGameDistribution where only the costs (of other players)
are varied."
abstract type AbstractCostDistribution <: AbstractGameDistribution end
function Base.rand(rng::AbstractRNG, d::AbstractCostDistribution)
    g = nominal_game(d)
    cost = map(player_costs(g), eachindex(player_costs(g))) do c, i
        # only modify cost of *other* players, ego cost is known
        return i > 1 ? rand(rng, d, c) : c
    end
    return @set(g.cost = cost)
end
function support(d::AbstractCostDistribution)
    g = nominal_game(d)
    # get all the possible costs for each player
    potential_player_costs = map(player_costs(g), eachindex(player_costs(g))) do c, i
        return i > 1 ? support(d, c) : [c]
    end

    # get all possible combinations
    return map(Iterators.product(potential_player_costs...)) do cost
        @set(g.cost = cost)
    end
end

"Cost distribution where the proximity cost of other players are varied."
struct ProximityCostDistribution{TG<:GeneralGame,TWD<:SparseCat} <: AbstractCostDistribution
    g::TG
    weight_distribution::TWD
    singular::Bool
end
function ProximityCostDistribution(g, w; p=[1/length(w) for _ in w], singular=false)
    return ProximityCostDistribution(g, SparseCat(w, p), singular)
end
nominal_game(d::ProximityCostDistribution) = d.g
issingulargame(d::ProximityCostDistribution) = d.singular
function cost_from_params(d::ProximityCostDistribution, c::NPlayerUnicycleCost,
                          w::Real)
    np = n_players(nominal_game(d))
    # same aggressiveness towards all other players
    ws = SVector{np}([w for i in 1:np])
    return @set(c.proximitycost.ws = ws)
end
function Base.rand(rng::AbstractRNG, d::ProximityCostDistribution,
                   c::NPlayerUnicycleCost)
    return cost_from_params(d, c, rand(rng, d.weight_distribution))
end
function support(d::ProximityCostDistribution, c::NPlayerUnicycleCost)
    return [cost_from_params(d, c, w) for w in d.weight_distribution.vals]
end
function game_info(d::ProximityCostDistribution, g::GeneralGame)
    return [first(c.proximitycost.ws) for c in player_costs(g)]
end

function isoptimistic(d::ProximityCostDistribution)
    vs = d.weight_distribution.vals
    return length(vs) == 1 && first(vs) >= 40
end
function ispessimistic(d::ProximityCostDistribution)
    vs = d.weight_distribution.vals
    return length(vs) == 1 && first(vs) <=  10
end
function isnominal(d::ProximityCostDistribution)
    return length(d.weight_distribution.vals) == 1 && !isoptimistic(d) && !ispessimistic(d)
end

"Cost distribution where the goals of other players are varied around a nominal goal"
struct GoalCostDistribution{TG,TGDA} <: AbstractCostDistribution
    g::TG
    Δϕs::TGDA
    singular::Bool
    function GoalCostDistribution(g::TG, Δϕs::TGDA=(0, pi/3); singular=false) where {TG,TGDA}
        return new{TG,TGDA}(g, Δϕs, singular)
    end
end
nominal_game(d::GoalCostDistribution) = d.g
issingulargame(d::GoalCostDistribution) = d.singular
function cost_from_params(::GoalCostDistribution, c::NPlayerUnicycleCost, Δϕ::Real)
    function rotate(xg, Δϕ)
        ϕ = xg[3] + Δϕ
        l = norm(xg[1:2])
        return SVector(l*cos(ϕ), l*sin(ϕ), ϕ, xg[4])
    end
    return @set c.goalcost.xg = rotate(c.goalcost.xg, Δϕ)
end
function Base.rand(rng::AbstractRNG, d::GoalCostDistribution, c::NPlayerUnicycleCost)
    return cost_from_params(d, c, rand(rng, d.Δϕs))
end
function support(d::GoalCostDistribution, c::NPlayerUnicycleCost)
    return [cost_from_params(d, c, Δϕ) for Δϕ in d.Δϕs]
end
function game_info(d::GoalCostDistribution, g::GeneralGame)
    return [c.goalcost.xg for c in player_costs(g)]
end

"------------------------ Game Solution State Distribution ------------------------"

struct GSSDistribution{TSD<:AbstractSeedDistribution, TOD<:AbstractGameDistribution}
    seed_distribution::TSD
    game_distribution::TOD
end
# TODO implement properly
issingulargame(d::GSSDistribution) = issingulargame(d.game_distribution)
issingularseed(d::GSSDistribution) = false
nominal_game(d::GSSDistribution) = nominal_game(d.game_distribution)
function Base.rand(rng::AbstractRNG, d::GSSDistribution)
    # op = zero_op(g))
    seed = rand(rng, d.seed_distribution)
    game_instance = rand(rng, d.game_distribution)
    return (seed, game_instance)
end
