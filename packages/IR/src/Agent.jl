# Agent definition
struct Agent{N,A,B,C,SA<:SArray{NTuple{N,2},A}}
    rule::SA
    ε::B
    α::C
    function Agent(rule::SA, ε::B, α::C) where {N,A,B,C,SA<:SArray{NTuple{N,2},A}}
        return new{N,A,B,C,SA}(rule, ε, α)
    end
end

"""
    (a::Agent)(info::SVector)

Calling an Agent (`a`) tells it to perceive the information (`info`) given, determine its
action, and execute it. The perception and execution parts of this process are
subject to errors `a.ε` and `a.α`.
"""
function (a::Agent)(info::SVector)
    return (Fix1(mistake, a.ε) ∘ Fix1(lerp, a.rule) ∘ Fix1(mistake, a.α))(info)
end
(a::Agent)(info...) = a(SA[info...])
# (a::Agent)(info::SVector) = execution_oopsie(a.ε, lerp(a.rule, mistake(a.α, info)))

function (a::Agent{3})(info::SVector)
    return (Fix1(mistake, a.ε) ∘ Fix1(lerp, a.rule) ∘ Fix1(mistake, a.α))(info)
end

function (a::Agent{2})(info::SVector)
    return (Fix1(execution_oopsie, a.ε) ∘ Fix1(lerp, a.rule) ∘ Fix1(mistake, a.α))(info)
end

Judge = Agent{3}
Player = Agent{2}

# relation first, reputation second, action third
# [rel, rep, act]

#TODO: when/why is agent(1, 0) slower than agent(true, false)?
