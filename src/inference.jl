"Generic external state update."
function update_external!(s, x)
    op = operating_point(s)
    Δx = x - first(op.x)
    for k in eachindex(op.x)
        op.x[k] += Δx
    end
    return s
end

struct GSS{TO,TG,TGA}
    "The initial operating point."
    op::TO
    "The initial strategy."
    γ::TG
    "The game instance (including the objectives etc.)"
    g::TGA
end

operating_point(gss::GSS) = gss.op
strategy(gss::GSS) = gss.γ
external(gss::GSS) = first(gss.op.x)
game_instance(gss::GSS) = gss.g

"----------------- a caching version of the state representation -----------------"

# TODO caching does not work like this anymore, unless the games also have the same
# objectives
"""
    $(TYPEDEF)

A caching database to avoid redundant computation of transitions.
"""
struct GSSCache{n, TS<:SizedVector{n, <:GSS}, TB<:SizedVector{n, Bool}}
    "The cached state (updated or not)."
    gss_vec::TS
    "A vector of Bools, element true if transition has been computed."
    internal_updated::TB
    "A vector of Bools, element true if the external state has been computed."
    external_updated::TB
end

function GSSCache(gss_vec)
    n = length(gss_vec)
    internal_updated = SizedVector{n}(falses(n))
    external_updated = SizedVector{n}(falses(n))
    return GSSCache(gss_vec, internal_updated, external_updated)
end

Base.length(::GSSCache{n}) where {n} = n
internal_updated(gss_cache::GSSCache, id::Int) = gss_cache.internal_updated[id]
external_updated(gss_cache::GSSCache, id::Int) = gss_cache.external_updated[id]
statelabel(gss_cache::GSSCache, id::Int) = gss_cache.gss_vec[id]
external(gss_cache::GSSCache, id::Int) = external(statelabel(gss_cache, id))
function reset_updated!(gss_cache::GSSCache)
    fill!(gss_cache.internal_updated, false)
    fill!(gss_cache.external_updated, false)
end

struct CachingGSS{TC<:GSSCache}
    "A reference to the cache."
    gss_cache::TC
    "The identifier for this state."
    id::Int
end

internal_updated(s::CachingGSS) = internal_updated(s.gss_cache, s.id)
external_updated(s::CachingGSS) = external_updated(s.gss_cache, s.id)
function update_external!(s::CachingGSS, x)
    if !external_updated(s)
        # call generic version
        s.gss_cache.external_updated[s.id] = true
        return invoke(update_external!, Tuple{Any, Any}, s, x)
    end
    return s
end
statelabel(s::CachingGSS) = statelabel(s.gss_cache, s.id)
external(s::CachingGSS) = external(statelabel(s))
operating_point(s::CachingGSS) = operating_point(statelabel(s))
strategy(s::CachingGSS) = strategy(statelabel(s))
game_instance(s::CachingGSS) = game_instance(statelabel(s))

"-------------------------------- POMDP interface --------------------------------"

function Base.circshift!(a::AbstractVector, shift::Integer)
    n = length(a)
    s = mod(-shift, n)
    s == 0 && return a
    reverse!(a, 1, s)
    reverse!(a, s+1, n)
    reverse!(a)
end

function shift_seed!(s, m)
    op = operating_point(s)
    γ = strategy(s)
    g = game_instance(s)

    # shift the strategy and set the last element to zero
    circshift!(γ, -1)
    γ[end] = zero(eltype(γ))

    # shift controls and set last control to zero
    circshift!(op.u, -1)
    op.u[end] = zero(eltype(op.u))

    # shift states and integrate the last step
    circshift!(op.x, -1)
    # Notice: don't be surprised that we take the last index here. The time
    # coordinate system of op is not shifted yet
    t_pre_end = time_disc2cont(op, length(op.x))
    op.x[end] = next_x(dynamics(g), op.x[end-1], op.u[end-1], t_pre_end)

    # the shifted time
    t0_shifted = initialtime(op) + samplingtime(op)
    # construct the shifted operating point
    op_shifted = SystemTrajectory{samplingtime(op)}(op.x, op.u, t0_shifted)
    # return the shifted state
    return GSS(op_shifted, γ, g)
end

struct GameSolutionModel{S,A,O,TS<:iLQSolver,TOM} <: POMDP{S,A,O}
    "The solver used for solving the game."
    solver::TS
    "Models the projection of a external(s) to the observation. Should be a callable
    `f(o::typeof(external(s)))`"
    observation_model::TOM
end

function GameSolutionModel{S,A,O}(solver::TS, om::TOM=identity) where {S,A,O,TS,TOM}
    return GameSolutionModel{S,A,O,TS,TOM}(solver, om)
end

solver(m::GameSolutionModel) = m.solver
observation_model(m::GameSolutionModel) = m.observation_model

function solve_with_fallback!(s, m::GameSolutionModel)
    x0 = external(s)
    op = operating_point(s)
    γ = strategy(s)
    g = game_instance(s)
    converged, _= solve!(op, γ, g, solver(m), x0)
    if !converged
        # if the solution did not converge, resolve it with the fallback seed
        zero!(op)
        fill!(γ, zero(eltype(γ)))
        converged, _ = solve!(op, γ, g, solver(m), x0)
    end
    return s
end

function gen_seed!(s, m::GameSolutionModel)
    s = solve_with_fallback!(s, m)
    s = shift_seed!(s, m)
    return s
end

gen_game!(s, m::GameSolutionModel) = s

function POMDPs.gen(m::GameSolutionModel, s::GSS, a::Float64, rng::AbstractRNG)
    s = gen_seed!(s, m)
    s = gen_game!(s, m)
    return (sp=s,)
end
"""
    $(TYPEDSIGNATURES)

In our case, the transition propagates the physical state according to the game
dynamics and uses a version of the seed. The transition does not depend
on the action, since in this setting we are a pure observer of the scene.
"""
function POMDPs.gen(m::GameSolutionModel, s::CachingGSS, a::Float64,
                    rng::AbstractRNG)
    # only compute transition if this state has not yet been upated
    if !internal_updated(s)
        s.gss_cache.gss_vec[s.id] = gen(m, statelabel(s), a, rng).sp
        s.gss_cache.internal_updated[s.id] = true
    end
    return (sp=s,)
end

"""
    $(TYPEDSIGNATURES)

The observation model that provides a likelihoood for an obsrvation `o` given the
state transition (s, a , sp).
"""
function POMDPModelTools.obs_weight(m::GameSolutionModel, sp, o)
    om = observation_model(m)
    # TODO: think about a more resonable Σ here and make part of model
    obs_σ² = 0.03^2
    return pdf(MvNormal(om(external(sp)), obs_σ²*(n_players(game_instance(sp)))I), o)
end

"---------------------------------- Belief Tools ----------------------------------"

gsscache(b::AbstractParticleBelief{<:CachingGSS}) = first(particles(b)).gss_cache

function merge_similar!(b::AbstractParticleBelief{<:CachingGSS},
                        max_elwise_diff::Real)
    ps = particles(b)
    for i in 1:(length(ps)-1), j in (i+1):length(ps)
        p1 = ps[i]; p2 = ps[j]
        if p1.id != p2.id && are_close(operating_point(p1), operating_point(p2), max_elwise_diff)
            ps[i] = ps[j]
        end
    end
end

reset_updated!(b::AbstractParticleBelief{<:CachingGSS}) = reset_updated!(gsscache(b))

function update_external!(b::AbstractParticleBelief{<:CachingGSS}, o)
    for p in particles(b)
        update_external!(p, o)
    end

    return b
end

# TODO: maybe extend ParticleFilters.mode instead
function sorted_mode(b::ParticleCollection{<:CachingGSS})
    ps = particles(b)
    counter = zeros(Int, length(gsscache(b)))

    for p in particles(b)
        counter[p.id] += 1
    end

    return particle(b, argmax(counter))
end
