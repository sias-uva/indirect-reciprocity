using StaticArrays
using JuMP
using Plots
import Ipopt
using Random

Random.seed!(11_04_2023)

# include("src.jl")


judge = (SA[1, 1, 0, 0, 0, 0, 1, 1], (0, (0, 0, 0)))
player = (SA[0, 0, 1, 1], (0.01, (0, 0)))

is_ESS(judge, player, player, 0.9, (2, 2, 1, 1))

# Globals:
benefit_sum_max = 4
cost_sum_min = 2

# Solver arguments:
## Social rules
global_n = SA[1, 1, 0, 0, 1, 1, 1, 1]
global_rs = SA[0, 0, 1, 1]
global_bs = SA[0, 0, 1, 1]
## IR model parameters:
global_pR = 0.9 # 1
global_u = (2, 2, 1, 1) # 4
## Mistake probabilities
global_jm = (0, (0, 0, 0)) # 4, (6, (7, 8, 9))
global_rm = (0.01, (0, 0)) # 3, (10, (11, 12))
global_bm = (0.01, (0, 0)) # 3, (13, (14, 15))
# total = 15

# Solver options:
domain_eps = 0.00001

# Objective function
function avg_payoffs(xs...)
    norm = SVector(xs[1:8])
    red_strategy = SVector(xs[9:12])
    blue_strategy = SVector(xs[13:16])
    utilities = xs[17:20]
    jem, jpm... = xs[21:24]
    judge_mistakes = (jem, jpm)
    judge = (norm, judge_mistakes)
    rem, rpm... = xs[25:27]
    red_mistakes = (rem, rpm)
    red = (red_strategy, red_mistakes)
    bem, bpm... = xs[28:30]
    blue_mistakes = (bem, bpm)
    blue = (blue_strategy, blue_mistakes)
    proportion_incumbents_red = xs[31]
    R★, B★ = stationary_incumbent_reputations(judge, red, blue, proportion_incumbents_red)
    avg = sum(incumbent_payoffs(red, blue, R★, B★, proportion_incumbents_red, utilities)) / 2
    return avg
end

# Constraint
function ESS_constraint(xs...)
    norm = SVector(xs[1:8])
    red_strategy = SVector(xs[9:12])
    blue_strategy = SVector(xs[13:16])
    utilities = xs[17:20]
    jem, jpm... = xs[21:24]
    judge_mistakes = (jem, jpm)
    judge = (norm, judge_mistakes)
    rem, rpm... = xs[25:27]
    red_mistakes = (rem, rpm)
    red = (red_strategy, red_mistakes)
    bem, bpm... = xs[28:30]
    blue_mistakes = (bem, bpm)
    blue = (blue_strategy, blue_mistakes)
    proportion_incumbents_red = xs[31]
    if is_ESS(judge, red, blue, proportion_incumbents_red, utilities)
        return true
    else
        invader(judge, red, blue, proportion_incumbents_red, utilities)
    end
end
ags = [global_n..., global_rs..., global_bs..., global_u..., global_jm[1], global_jm[2]..., global_rm[1], global_rm[2]..., global_bm[1], global_bm[2]..., global_pR]

function payoff_red(xs...)
    norm = SVector(xs[1:8])
    red_strategy = SVector(xs[9:12])
    blue_strategy = SVector(xs[13:16])
    utilities = xs[17:20]
    jem, jpm... = xs[21:24]
    judge_mistakes = (jem, jpm)
    judge = (norm, judge_mistakes)
    rem, rpm... = xs[25:27]
    red_mistakes = (rem, rpm)
    red = (red_strategy, red_mistakes)
    bem, bpm... = xs[28:30]
    blue_mistakes = (bem, bpm)
    blue = (blue_strategy, blue_mistakes)
    proportion_incumbents_red = xs[31]
    R★, B★ = stationary_incumbent_reputations(judge, red, blue, proportion_incumbents_red)
    payoff_red, _ = incumbent_payoffs(red, blue, R★, B★, proportion_incumbents_red, utilities)
    return payoff_red
end

function payoff_blue(xs...)
    norm = SVector(xs[1:8])
    red_strategy = SVector(xs[9:12])
    blue_strategy = SVector(xs[13:16])
    utilities = xs[17:20]
    jem, jpm... = xs[21:24]
    judge_mistakes = (jem, jpm)
    judge = (norm, judge_mistakes)
    rem, rpm... = xs[25:27]
    red_mistakes = (rem, rpm)
    red = (red_strategy, red_mistakes)
    bem, bpm... = xs[28:30]
    blue_mistakes = (bem, bpm)
    blue = (blue_strategy, blue_mistakes)
    proportion_incumbents_red = xs[31]
    R★, B★ = stationary_incumbent_reputations(judge, red, blue, proportion_incumbents_red)
    _, payoff_blue = incumbent_payoffs(red, blue, R★, B★, proportion_incumbents_red, utilities)
    return payoff_blue
end

for i in 0:15
    mr = SVector{4,Bool}(digits(i, base=2, pad=4))
    red_name = Symbol(string("payoff_red_mutant_", i))
    blue_name = Symbol(string("payoff_blue_mutant_", i))
    mutant_rule = SVector{4,Bool}((false, false, false, false))
    @eval begin
        function $(red_name)(xs...)
            norm = SVector(xs[1:8])
            red_strategy = SVector(xs[9:12])
            blue_strategy = SVector(xs[13:16])
            utilities = xs[17:20]
            jem, jpm... = xs[21:24]
            judge_mistakes = (jem, jpm)
            judge = (norm, judge_mistakes)
            rem, rpm... = xs[25:27]
            red_mistakes = (rem, rpm)
            red = (red_strategy, red_mistakes)
            bem, bpm... = xs[28:30]
            blue_mistakes = (bem, bpm)
            blue = (blue_strategy, blue_mistakes)
            proportion_incumbents_red = xs[31]
            mutant_rule = $(mr)
            red_mutant = (mutant_rule, red_mistakes)
            blue_mutant = (mutant_rule, blue_mistakes)
            R★, B★, RM★, BM★ = stationary_reputations(judge, red, blue, red_mutant, blue_mutant, proportion_incumbents_red)
            payoff_red_mutant, _ = mutant_payoffs(red, blue, red_mutant, blue_mutant, R★, B★, RM★, BM★, proportion_incumbents_red, utilities)
            return payoff_red_mutant
        end
        function $(blue_name)(xs...)
            norm = SVector(xs[1:8])
            red_strategy = SVector(xs[9:12])
            blue_strategy = SVector(xs[13:16])
            utilities = xs[17:20]
            jem, jpm... = xs[21:24]
            judge_mistakes = (jem, jpm)
            judge = (norm, judge_mistakes)
            rem, rpm... = xs[25:27]
            red_mistakes = (rem, rpm)
            red = (red_strategy, red_mistakes)
            bem, bpm... = xs[28:30]
            blue_mistakes = (bem, bpm)
            blue = (blue_strategy, blue_mistakes)
            proportion_incumbents_red = xs[31]
            mutant_rule = $(mr)
            red_mutant = (mutant_rule, red_mistakes)
            blue_mutant = (mutant_rule, blue_mistakes)
            R★, B★, RM★, BM★ = stationary_reputations(judge, red, blue, red_mutant, blue_mutant, proportion_incumbents_red)
            _, payoff_blue_mutant = mutant_payoffs(red, blue, red_mutant, blue_mutant, R★, B★, RM★, BM★, proportion_incumbents_red, utilities)
            return payoff_blue_mutant
        end
    end
end

avg_payoffs(ags...)
ESS_constraint(ags...)
payoff_blue(ags...)
payoff_red(ags...)
payoff_red_mutant_15(ags...)
payoff_red_mutant_12(ags...)
payoff_red_mutant_8(ags...)
payoff_red_mutant_1(ags...)
payoff_red_mutant_0(ags...)

# Constraints:
begin
    model = Model(Ipopt.Optimizer)
    register(model, :avg_payoffs, 31, avg_payoffs; autodiff=true)
    register(model, :payoff_red, 31, payoff_red; autodiff=true)
    register(model, :payoff_blue, 31, payoff_blue; autodiff=true)
    @variable(model, domain_eps <= norm[1:8] <= 1-domain_eps)
    @variable(model, domain_eps <= red_strategy[1:3] <= 1-domain_eps)
    @variable(model, domain_eps <= blue_strategy[1:3] <= 1-domain_eps)
    @variable(model, domain_eps <= utilities[1:4] <= Inf)
    @variable(model, domain_eps <= mistakes[1:10] <= 0.01, start = 0.005)
    @variable(model, 0.5 <= pR <= 0.9)
    vars = (norm..., red_strategy..., 1, blue_strategy..., 1, utilities..., mistakes..., pR)
    @NLobjective(model, Max, avg_payoffs(vars...))
    for i in 0:15
        print(i)
        red_name = Symbol(string("payoff_red_mutant_", i))
        blue_name = Symbol(string("payoff_blue_mutant_", i))
        @eval begin
            register(model, Symbol(string("payoff_red_mutant_", $i)), 31, $(red_name); autodiff=true)
            register(model, Symbol(string("payoff_blue_mutant_", $i)), 31, $(blue_name); autodiff=true)
            @NLconstraint(model, payoff_red(vars...) >= $(red_name)(vars...))
            @NLconstraint(model, payoff_blue(vars...) >= $(blue_name)(vars...))
        end
    end
    @NLconstraint(model, 0 <= utilities[1] + utilities[2] <= benefit_sum_max)
    @NLconstraint(model, utilities[3] + utilities[4] >= cost_sum_min)
    @NLconstraint(model, utilities[1] == utilities[2])
    @NLconstraint(model, utilities[3] == utilities[4])
    set_optimizer_attribute(model, "max_iter", 3000)
end

optimize!(model)
objective_value(model)
value(pR)
value.(utilities)
value.(mistakes)
value.(norm)
value.(red_strategy)
value.(blue_strategy)