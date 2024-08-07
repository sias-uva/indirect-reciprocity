using IR
using Base: Fix1
using StaticArrays

struct InterpolatingNorm{T}
    n::SArray{NTuple{2,2},T}
    InterpolatingNorm(va::Vararg{T,4}) where {T} = new{T}(SArray{NTuple{2,2},T}(va...))
end

function (interp_norm::InterpolatingNorm)(action, opponent_reputation)
    return lerp(interp_norm.n, SA[opponent_reputation, action])
end

struct ThresholdAgent{T}
    θ::T
    ε::Float64
    α::Float64
    ThresholdAgent(t::T, e, a) where {T} = new{T}(t, e, a)
end

function (ta::ThresholdAgent)(opponent_reputation)
    return (Fix1(mistake, ta.ε) ∘ >(ta.θ) ∘ Fix1(mistake, ta.α))(opponent_reputation)
end

stern_judging = InterpolatingNorm(1, -1, -1, 1)
image_score = InterpolatingNorm(-1, -1, 1, 1)
continuation = InterpolatingNorm(-1, -1, -1, 1)

ta = ThresholdAgent(0.5, 0.1, 0)

function new_reputation(norm, reputation, action, opponent_reputation)
    rep_update = norm(action, opponent_reputation)
    println("Norm says $(rep_update >= 0 ? "+" : "")$rep_update")
    return reputation + rep_update
end

n_agents = 100
reputations = ones(n_agents)
agents = [ThresholdAgent(0.5, 0.1, 0) for _ in 1:n_agents]

for (i, ta) in enumerate(agents)
    # pick random opponent
    j, other_ta = let
        candidate_j = rand(1:(n_agents - 1))
        j = candidate_j < i ? candidate_j : candidate_j + 1
        j, agents[j]
    end
    # update reputation after playing
    nr = new_reputation(continuation, reputations[i], ta(reputations[j]), reputations[j])
    println(nr)
    reputations[i] = clamp(nr, 0, 1)
end

reputations

new_reputation(continuation, 1, 0.1, 1)

ta(1)
