using Optimization
using OptimizationMOI
using Ipopt
using OptimizationOptimJL
using IR
using StaticArrays

function unpack_x(x)
    jem, jpm... = x[1:4]
    rem, rpm... = x[5:7]
    bem, bpm... = x[8:10]
    utilities = x[11:14]
    pR = x[15]
    return jem, SA[jpm...], rem, SA[rpm...], bem, SA[bpm...], pR, utilities
end

function both_payoffs(jem, jpm, rem, rpm, bem, bpm, pR, utilities; norm, red_strategy, blue_strategy)
    judge = Agent(norm, jem, jpm)
    red = Agent(red_strategy, rem, rpm)
    blue = Agent(blue_strategy, bem, bpm)
    R★, B★ = stationary_incumbent_reputations(judge, red, blue, pR)
    return incumbent_payoffs(red, blue, R★, B★, pR, utilities)
end

function avg_payoffs(x, p)
    jem, jpm, rem, rpm, bem, bpm, pR, utilities = unpack_x(x)
    norm, red_strategy, blue_strategy = p
    red_payoff, blue_payoff = both_payoffs(jem, jpm, rem, rpm, bem, bpm, pR, utilities; norm, red_strategy, blue_strategy)
    return (red_payoff + blue_payoff)/2
end

function cons(res, x, p)
    jem, jpm, rem, rpm, bem, bpm, pR, utilities = unpack_x(x)
    norm, red_strategy, blue_strategy = p
    judge = Agent(norm, jem, jpm)
    red = Agent(red_strategy, rem, rpm)
    blue = Agent(blue_strategy, bem, bpm)
    
    R★, B★ = stationary_incumbent_reputations(judge, red, blue, pR)
    red_payoff, blue_payoff = incumbent_payoffs(red, blue, R★, B★, pR, utilities)
    # ESS constraints
    for i in 1:16
        mutant_strategy = iStrategy(i-1)
        red_mutant = Agent(mutant_strategy, rem, rpm)
        blue_mutant = Agent(mutant_strategy, bem, bpm)
        RM★, RB★ = stationary_mutant_reputations(judge, red, blue, R★, B★, pR)
        red_mutant_payoff, blue_mutant_payoff = mutant_payoffs(red, blue, red_mutant, blue_mutant, R★, B★, RM★, RB★, pR, utilities)
        res[2*i-1] = red_payoff - red_mutant_payoff
        res[2*i] = blue_payoff - blue_mutant_payoff
    end
    # Utilities must be constrained
    res[33] = utilities[1] + utilities[2] # (Upper bound) sum of benefit
    res[34] = utilities[3] + utilities[4] # (Lower bound) sum of cost
    # Utilities must be equal
    res[35] = utilities[1] - utilities[2]
    res[36] = utilities[3] - utilities[4]
end

variable_names = [:jem, :jpm_rel, :jpm_rep, :jpm_act, :rem, :rpm_rel, :rpm_rep, :bem, :bpm_rel, :rpm_rep, :pR, :benefit_red, :benefit_blue, :cost_red, :cost_blue]
parameter_names = [:norm, :red_strategy, :blue_strategy]
res = zeros(30)
nss_combination = (iNorm(195), iStrategy(12), iStrategy(12))

constraint_parameters = (
    benefit_sum_max = 4, # Sum of each group's benefit
    cost_sum_min = 2, # Sum of each group's costs
    em_bound = 0.5,
    pm_bound = 0.4,
)

f = OptimizationFunction(avg_payoffs, Optimization.AutoForwardDiff(); cons = cons, syms = variable_names, paramsyms = parameter_names)

# Just over/under bounds to ensure initial point is an interior point (Slater's cond I guess?)
initial_errors = repeat([0.01], 10)
initial_utilities = (1.9, 1.9, 1.1, 1.1) 
initial_prob_red = 0.8

x0 = [initial_errors..., initial_prob_red, initial_utilities...]

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

prob = OptimizationProblem(f, x0, nss_combination; lb, ub, lcons, ucons, sense = MaxSense)
# solve(prob, IPNewton())
sol = solve(prob, Ipopt.Optimizer()) # Always returns 99999? Completely disregarding the bounds on variables.
sol[:cost_blue]