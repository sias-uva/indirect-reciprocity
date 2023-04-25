module IR

using StaticArrays
using Base: Fix1, Fix2
using LinearAlgebra: I, det

export mistake, lerp
export Agent, Judge, Player, Norm, Strategy, iNorm, iStrategy
export stationary_incumbent_reputations, stationary_mutant_reputations, stationary_reputations
export p_donates, p_receives, incumbent_payoffs, mutant_payoffs, payoffs
export is_ESS, invader

# types
iNorm(i::Integer) = Norm(i >> shift & 1 != 0 for shift in 0:7)
iStrategy(i::Integer) = Strategy(i >> shift & 1 != 0 for shift in 0:3)
Norm = SArray{NTuple{3,2}, Bool}
Strategy = SArray{NTuple{2,2}, Bool}

include("nlinear_interpolation.jl")
include("Agent.jl")
include("reputations.jl")
include("payoffs.jl")

end # module