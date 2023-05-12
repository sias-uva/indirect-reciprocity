module RL

export LearningAgent

using IR
using Agents
using StaticArrays
using Base: Fix1

@agent LearningAgent{P,E,A,G} NoSpaceAgent begin
    policy::P
    ε::E
    α::A
    group::G
    exploration_probability::Float64
end

LearningAgent(policy::P, ε::E, α::A, group::G, whatever...) where {P,E,A,G} = LearningAgent{P,E,A,G}(policy, ε, α, group, whatever...)

function (la::LearningAgent)(info::SVector)
    if rand() < la.exploration_probability
        return rand(Bool)
    end
    exploit = (Fix1(mistake, la.ε) ∘ Fix1(evaluate, la.policy) ∘ Fix1(mistake, la.α))(info)
    return rand() < exploit
end
(la::LearningAgent)(info...) = la(SA[info...])

function evaluate(policy::P, info) where {T,P<:SArray{Tuple{2,2,2},T}}
    strategy = SArray{NTuple{2,2},T,2,4}(maximum(policy, dims=3))
    return lerp(strategy, info)
end

end # module

# Difference: Learning on all fronts
# Make use of the uncertainty in information