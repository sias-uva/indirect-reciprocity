module IR

using StaticArrays
using Base: Fix1, Fix2
using LinearAlgebra: I, det

export mistake, lerp
export Agent, Judge, Player, Norm, Strategy, iNorm, iStrategy
export stationary_incumbent_reputations,
    stationary_mutant_reputations, stationary_reputations
export p_donates, p_receives, incumbent_payoffs, mutant_payoffs, payoffs
export is_ESS, invader

# Types
iNorm(i::Integer) = Norm(i >> shift & 1 != 0 for shift in 0:7)
iStrategy(i::Integer) = Strategy(i >> shift & 1 != 0 for shift in 0:3)
Norm = SArray{NTuple{3,2},Bool}
Strategy = SArray{NTuple{2,2},Bool}

include("nlinear_interpolation.jl")
include("Agent.jl")
include("reputations.jl")
include("payoffs.jl")

# Precompilation
using PrecompileTools
@setup_workload begin
    p = (;
        rem=0.01,
        bem=0.01,
        jem=0.01,
        rpm=SA[0.0,0.0],
        bpm=SA[0.0,0.0],
        jpm=SA[0.0, 0.0, 0.0],
        pR=0.9,
        utilities=SA[2, 2, 1, 1],
    )
    @compile_workload begin
        norm = iNorm(195)
        judge = Agent(norm, p.jem, p.jpm)
        red_strat = iStrategy(12)
        red = Agent(red_strat, p.rem, p.rpm)
        blue_strat = iStrategy(12)
        blue = Agent(blue_strat, p.bem, p.bpm)
        is_ESS(judge, red, blue, p.pR, p.utilities)
    end
end

end # module
