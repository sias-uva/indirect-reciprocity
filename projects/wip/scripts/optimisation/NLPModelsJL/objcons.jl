using IR
using IR: Norm, Strategy
using StaticArrays
using ObjConsNLPModels, Percival, NLPModels

# struct ArbitraryCallable{T} <: Function
#     iv::T
# end

# function (ac::ArbitraryCallable{T})(x) where {T}
#     obj = sum(abs2, x)
#     cons = sum(x) - ac.iv
#     return [obj, cons]
# end
# ac = ArbitraryCallable{Float64}(1)

# model = objcons_nlpmodel(ac; x0=[2.0, 2.0])

tmodel = objcons_nlpmodel(x -> [sum(abs2, x), sum(x) - 1]; x0=[2.0, 2.0])
output = percival(tmodel)
output.solution
output.objective

struct IRSystem{T,N<:SArray{NTuple{3,2},T},S<:SArray{NTuple{2,2},T}}
    norm::N
    majority_strategy::S
    minority_strategy::S
    majority_mutant_strategies::Vector{S}
    minority_mutant_strategies::Vector{S}
end

function IRSystem(n, r, b)
    all_strategies = iStrategy.(0:15)
    S = typeof(r)
    mrs = Vector{S}(setdiff(all_strategies, [r]))
    mbs = Vector{S}(setdiff(all_strategies, [b]))
    return IRSystem(n, r, b, mrs, mbs)
end

norm::Norm = iNorm(195)
majority_strategy::Strategy = iStrategy(12)
minority_strategy::Strategy = iStrategy(12)

irs = IRSystem(norm, majority_strategy, minority_strategy)

function (irs::IRSystem)(
    judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm, prop_majority, utilities
)
    judge = Agent(irs.norm, judge_em, judge_pm)
    majority = Agent(irs.majority_strategy, maj_em, maj_pm)
    minority = Agent(irs.minority_strategy, min_em, min_pm)
    majority_rep, minority_rep = stationary_incumbent_reputations(
        judge, majority, minority, prop_majority
    )
    incumbent_payoff_values = incumbent_payoffs(
        majority, minority, majority_rep, minority_rep, prop_majority, utilities
    )
    _cons = map(
        zip(irs.majority_mutant_strategies, irs.minority_mutant_strategies)
    ) do (rms, bms)
        majority_mutant = Agent(rms, maj_em, maj_pm)
        minority_mutant = Agent(bms, min_em, maj_pm)
        majority_mutant_rep, minority_mutant_rep = stationary_mutant_reputations(
            judge,
            majority_mutant,
            minority_mutant,
            majority_rep,
            minority_rep,
            prop_majority,
        )
        mutant_payoff_values = mutant_payoffs(
            majority,
            minority,
            majority_mutant,
            minority_mutant,
            majority_rep,
            minority_rep,
            majority_mutant_rep,
            minority_mutant_rep,
            prop_majority,
            utilities,
        )
        incumbent_payoff_values .- mutant_payoff_values
    end
    T = eltype(eltype(_cons))
    cons = reinterpret(T, _cons)
    obj = mean(incumbent_payoff_values)
    return [obj, cons...]
end

function (irs::IRSystem)(x)
    T = typeof(x[1])
    judge_em = x[1]
    judge_pm = SVector{3,T}(x[2:4])
    maj_em = x[5]
    maj_pm = SVector{2,T}(x[6:7])
    min_em = x[8]
    min_pm = SVector{2,T}(x[9:10])
    prop_majority = x[11]
    utilities = SVector{4,T}(x[12:15])
    return irs(judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm, prop_majority, utilities)
end

x0 = [
    0.0, SA[0, 0, 0.0]..., 0.01, SA[0, 0.0]..., 0.01, SA[0, 0.0]..., 0.8, SA[2, 2, 1, 1]...
]

constraint_parameters = (
    benefit_sum_max=4, # Sum of each group's benefit
    cost_sum_min=2, # Sum of each group's costs
    em_bound=0.5,
    pm_bound=0.4,
)

lvar = zeros(15)
uvar = let
    bounds = ones(15)
    ems = [1, 5, 8]
    pms = [2:4..., 6:7..., 9:10...]
    bounds[ems] .= constraint_parameters[:em_bound]
    bounds[pms] .= constraint_parameters[:pm_bound]
    bounds[12:15] .= Inf
    bounds
end

irs(x0)[1]
all(irs(x0)[2:end] .> 0)

model = objcons_nlpmodel(irs; x0=x0, lvar, uvar)
