# Agent definition
struct Agent{N,A,B,C,SA<:SArray{NTuple{N,2},A}}
    rule::SA
    ε::B
    α::C
    function Agent(rule::SA, ε::B, α::C) where {N,A,B,C,SA<:SArray{NTuple{N,2},A}}
        return new{N,A,B,C,SA}(rule, ε, α)
    end
end

function (a::Agent)(info::SVector)
    return (Fix1(mistake, a.ε) ∘ Fix1(lerp, a.rule) ∘ Fix1(mistake, a.α))(info)
end
(a::Agent)(info...) = a(SA[info...])
# (a::Agent)(info::SVector) = mistake(a.ε, lerp(a.rule, mistake(a.α, info)))

Judge = Agent{3}
Player = Agent{2}
