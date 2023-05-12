module RL

export LearningAgent # core
export whichmax, evaluate # misc

using IR
using Agents
using StaticArrays
using Base: Fix1

whichmax(x, y) = argmax((x, y))-1

@agent LearningAgent{P,E,A,G} NoSpaceAgent begin
    policy::P
    ε::E
    α::A
    group::G
    exploration_probability::Float64
end

LearningAgent(policy::P, ε::E, α::A, group::G, whatever...) where {P,E,A,G} = LearningAgent{P,E,A,G}(policy, ε, α, group, whatever...)

function (la::LearningAgent)(info::SVector)
    la.exploration_probability > rand() && return rand(Bool) # Explore
    prob_coop = (Fix1(mistake, la.ε) ∘ Fix1(evaluate, la.policy) ∘ Fix1(mistake, la.α))(info)
    return prob_coop > rand()
end
(la::LearningAgent)(info...) = la(SA[info...])

function evaluate(policy::P, info::SVector{2}) where {T,P<:SArray{Tuple{2,2,2},T}}
    strategy = SArray{NTuple{2,2},T,2,4}(reduce(whichmax, policy; dims=3))
    return lerp(strategy, info)
end

# Commenting this method out because it's no faster than the above due to a
# custom lerp implementation for 2×2 SArrays:

# function evaluate(policy::P, info::SVector{2, Bool}) where {T,P<:SArray{Tuple{2,2,2},T}}
#     strategy = SArray{NTuple{2,2},T,2,4}(reduce(whichmax, policy; dims=3))
#     return strategy[info[1]+1, info[2]+1]
# end

end # module

