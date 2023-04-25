module TinyIR

using StaticArrays

export evaluate
export stationary_incumbent_reputations, stationary_mutant_reputations, stationary_reputations
export p_donates, p_receives, incumbent_payoffs, mutant_payoffs, payoffs
export is_ESS, invader

include("evaluate.jl")
include("reputations.jl")
include("payoffs.jl")

end # module