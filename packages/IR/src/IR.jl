module IR

using StaticArrays
using Base: Fix1, Fix2
using LinearAlgebra: I, det

export mistake, execution_oopsie, lerp
export Agent, Judge, Player, iNorm, iStrategy
export stationary_incumbent_reputations,
    stationary_mutant_reputations, stationary_reputations
export p_donates, p_receives, incumbent_payoffs, mutant_payoffs, payoffs
export is_ESS, invader

# Types
iNorm(i::Integer) = Norm(i >> shift & 1 != 0 for shift in 0:7)
iStrategy(i::Integer) = Strategy(i >> shift & 1 != 0 for shift in 0:3)
const Norm = SArray{NTuple{3,2},Bool}
const Strategy = SArray{NTuple{2,2},Bool}

# iStrategy(i::Integer) = SArray{NTuple{2,2},Bool}(i >> shift & 1 != 0 for shift in 0:3)

include("nlinear_interpolation.jl")
include("Agent.jl")
include("reputations.jl")
include("payoffs.jl")

# Precompilation
using PrecompileTools
@setup_workload begin
    p = (;
        maj_em=0.01,
        min_em=0.01,
        judge_em=0.01,
        maj_pm=SA[0.0, 0.0],
        min_pm=SA[0.0, 0.0],
        judge_pm=SA[0.0, 0.0, 0.0],
        prop_maj=0.9,
        utilities=SA[2, 2, 1, 1],
    )
    @compile_workload begin
        norm = iNorm(195)
        judge = Agent(norm, p.judge_em, p.judge_pm)
        majority_strat = iStrategy(12)
        majority = Agent(majority_strat, p.maj_em, p.maj_pm)
        minority_strat = iStrategy(12)
        minority = Agent(minority_strat, p.min_em, p.min_pm)
        is_ESS(judge, majority, minority, p.prop_maj, p.utilities)
    end
end

end # module
