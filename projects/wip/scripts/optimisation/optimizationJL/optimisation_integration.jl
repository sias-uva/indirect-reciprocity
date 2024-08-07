using Optimization
using OptimizationMOI
using Ipopt
using OptimizationOptimJL
using IR
using StaticArrays

function unpack_x(x)
    judge_em, judge_pm... = x[1:4]
    maj_em, maj_pm... = x[5:7]
    min_em, min_pm... = x[8:10]
    utilities = x[11:14]
    prop_maj = x[15]
    return judge_em,
    SA[judge_pm...], maj_em, SA[maj_pm...], min_em, SA[min_pm...], prop_maj,
    utilities
end

function both_payoffs(
    judge_em,
    judge_pm,
    maj_em,
    maj_pm,
    min_em,
    min_pm,
    prop_maj,
    utilities;
    norm,
    majority_strategy,
    minority_strategy,
)
    judge = Agent(norm, judge_em, judge_pm)
    majority = Agent(majority_strategy, maj_em, maj_pm)
    minority = Agent(minority_strategy, min_em, min_pm)
    R★, B★ = stationary_incumbent_reputations(judge, majority, minority, prop_maj)
    return incumbent_payoffs(majority, minority, R★, B★, prop_maj, utilities)
end

function avg_payoffs(x, p)
    judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm, prop_maj, utilities = unpack_x(x)
    norm, majority_strategy, minority_strategy = p
    majority_payoff, minority_payoff = both_payoffs(
        judge_em,
        judge_pm,
        maj_em,
        maj_pm,
        min_em,
        min_pm,
        prop_maj,
        utilities;
        norm,
        majority_strategy,
        minority_strategy,
    )
    return (majority_payoff + minority_payoff) / 2
end

function cons(res, x, p)
    judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm, prop_maj, utilities = unpack_x(x)
    norm, majority_strategy, minority_strategy = p
    judge = Agent(norm, judge_em, judge_pm)
    majority = Agent(majority_strategy, maj_em, maj_pm)
    minority = Agent(minority_strategy, min_em, min_pm)

    R★, B★ = stationary_incumbent_reputations(judge, majority, minority, prop_maj)
    majority_payoff, minority_payoff = incumbent_payoffs(
        majority, minority, R★, B★, prop_maj, utilities
    )
    # ESS constraints
    for i in 1:16
        mutant_strategy = iStrategy(i - 1)
        majority_mutant = Agent(mutant_strategy, maj_em, maj_pm)
        minority_mutant = Agent(mutant_strategy, min_em, min_pm)
        RM★, RB★ = stationary_mutant_reputations(
            judge, majority, minority, R★, B★, prop_maj
        )
        majority_mutant_payoff, minority_mutant_payoff = mutant_payoffs(
            majority,
            minority,
            majority_mutant,
            minority_mutant,
            R★,
            B★,
            RM★,
            RB★,
            prop_maj,
            utilities,
        )
        res[2 * i - 1] = majority_payoff - majority_mutant_payoff
        res[2 * i] = minority_payoff - minority_mutant_payoff
    end
    # Utilities must be constrained
    res[33] = utilities[1] + utilities[2] # (Upper bound) sum of benefit
    res[34] = utilities[3] + utilities[4] # (Lower bound) sum of cost
    # Utilities must be equal
    res[35] = utilities[1] - utilities[2]
    return res[36] = utilities[3] - utilities[4]
end

variable_names = [
    :judge_em,
    :jpm_rel,
    :jpm_rep,
    :jpm_act,
    :maj_em,
    :rpm_rel,
    :rpm_rep,
    :min_em,
    :bpm_rel,
    :rpm_rep,
    :prop_maj,
    :benefit_majority,
    :benefit_minority,
    :cost_majority,
    :cost_minority,
]
parameter_names = [:norm, :majority_strategy, :minority_strategy]
res = zeros(30)
nss_combination = (iNorm(195), iStrategy(12), iStrategy(12))

constraint_parameters = (
    benefit_sum_max=4, # Sum of each group's benefit
    cost_sum_min=2, # Sum of each group's costs
    em_bound=0.5,
    pm_bound=0.4,
)

f = OptimizationFunction(
    avg_payoffs,
    Optimization.AutoForwardDiff();
    cons=cons,
    syms=variable_names,
    paramsyms=parameter_names,
)

# Just over/under bounds to ensure initial point is an interior point (Slater's cond I guess?)
initial_errors = repeat([0.01], 10)
initial_utilities = (1.9, 1.9, 1.1, 1.1)
initial_prob_majority = 0.8

x0 = [initial_errors..., initial_prob_majority, initial_utilities...]

lb = zeros(15)
ub = let
    bounds = ones(15)
    ems = [1, 5, 8]
    pms = [2:4..., 6:7..., 9:10...]
    bounds[ems] .= constraint_parameters[:em_bound]
    bounds[pms] .= constraint_parameters[:pm_bound]
    bounds[12:15] .= Inf
    bounds
end

lcons = let
    bounds = zeros(36)
    bounds[34] = constraint_parameters[:cost_sum_min]
    bounds
end
ucons = let
    bounds = zeros(36)
    bounds[1:32] .= Inf
    bounds[33] = constraint_parameters[:benefit_sum_max]
    bounds[34] = Inf
    bounds
end

prob = OptimizationProblem(f, x0, nss_combination; lb, ub, lcons, ucons, sense=MaxSense)
# solve(prob, IPNewton())
sol = solve(prob, Ipopt.Optimizer()) # Always returns 99999? Completely disregarding the bounds on variables.
sol[:cost_minority]
