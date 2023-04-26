using TinyIR

# Objective function
function avg_payoffs(xs...)
    proportion_incumbents_red = xs[1]
    utilities = xs[2:5]
    jem, jpm... = xs[6:9]
    judge_mistakes = (jem, jpm)
    judge = (norm, judge_mistakes)
    rem, rpm... = xs[10:12]
    red_mistakes = (rem, rpm)
    red = (red_strategy, red_mistakes)
    bem, bpm... = xs[13:15]
    blue_mistakes = (bem, bpm)
    blue = (blue_strategy, blue_mistakes)
    R★, B★ = stationary_incumbent_reputations(judge, red, blue, proportion_incumbents_red)
    avg = sum(incumbent_payoffs(red, blue, R★, B★, proportion_incumbents_red, utilities)) / 2
    return avg
end

# Helper function
function both_payoffs(xs...)
    proportion_incumbents_red = xs[1]
    utilities = xs[2:5]
    jem, jpm... = xs[6:9]
    judge_mistakes = (jem, jpm)
    judge = (norm, judge_mistakes)
    rem, rpm... = xs[10:12]
    red_mistakes = (rem, rpm)
    red = (red_strategy, red_mistakes)
    bem, bpm... = xs[13:15]
    blue_mistakes = (bem, bpm)
    blue = (blue_strategy, blue_mistakes)
    R★, B★ = stationary_incumbent_reputations(judge, red, blue, proportion_incumbents_red)
    return incumbent_payoffs(red, blue, R★, B★, proportion_incumbents_red, utilities)
end

# Constraint
function ESS_constraint(xs...)
    proportion_incumbents_red = xs[1]
    utilities = xs[2:5]
    jem, jpm... = xs[6:9]
    judge_mistakes = (jem, jpm)
    judge = (norm, judge_mistakes)
    rem, rpm... = xs[10:12]
    red_mistakes = (rem, rpm)
    red = (red_strategy, red_mistakes)
    bem, bpm... = xs[13:15]
    blue_mistakes = (bem, bpm)
    blue = (blue_strategy, blue_mistakes)
    if is_ESS(judge, red, blue, proportion_incumbents_red, utilities)
        return true
    else
        invader(judge, red, blue, proportion_incumbents_red, utilities)
    end
end
# ags = [proportion_incumbents_red, utils..., judge_mistakes[1], judge_mistakes[2]..., red_mistakes[1], red_mistakes[2]..., blue_mistakes[1], blue_mistakes[2]...]

function payoff_red(xs...)
    proportion_incumbents_red = xs[1]
    utilities = xs[2:5]
    jem, jpm... = xs[6:9]
    judge_mistakes = (jem, jpm)
    judge = (norm, judge_mistakes)
    rem, rpm... = xs[10:12]
    red_mistakes = (rem, rpm)
    red = (red_strategy, red_mistakes)
    bem, bpm... = xs[13:15]
    blue_mistakes = (bem, bpm)
    blue = (blue_strategy, blue_mistakes)
    R★, B★ = stationary_incumbent_reputations(judge, red, blue, proportion_incumbents_red)
    payoff_red, _ = incumbent_payoffs(red, blue, R★, B★, proportion_incumbents_red, utilities)
    return payoff_red
end

function payoff_blue(xs...)
    proportion_incumbents_red = xs[1]
    utilities = xs[2:5]
    jem, jpm... = xs[6:9]
    judge_mistakes = (jem, jpm)
    judge = (norm, judge_mistakes)
    rem, rpm... = xs[10:12]
    red_mistakes = (rem, rpm)
    red = (red_strategy, red_mistakes)
    bem, bpm... = xs[13:15]
    blue_mistakes = (bem, bpm)
    blue = (blue_strategy, blue_mistakes)
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
            proportion_incumbents_red = xs[1]
            utilities = xs[2:5]
            jem, jpm... = xs[6:9]
            judge_mistakes = (jem, jpm)
            judge = (norm, judge_mistakes)
            rem, rpm... = xs[10:12]
            red_mistakes = (rem, rpm)
            red = (red_strategy, red_mistakes)
            bem, bpm... = xs[13:15]
            blue_mistakes = (bem, bpm)
            blue = (blue_strategy, blue_mistakes)
            mutant_rule = $(mr)
            red_mutant = (mutant_rule, red_mistakes)
            blue_mutant = (mutant_rule, blue_mistakes)
            R★, B★, RM★, BM★ = stationary_reputations(judge, red, blue, red_mutant, blue_mutant, proportion_incumbents_red)
            payoff_red_mutant, _ = mutant_payoffs(red, blue, red_mutant, blue_mutant, R★, B★, RM★, BM★, proportion_incumbents_red, utilities)
            return payoff_red_mutant
        end
        function $(blue_name)(xs...)
            proportion_incumbents_red = xs[1]
            utilities = xs[2:5]
            jem, jpm... = xs[6:9]
            judge_mistakes = (jem, jpm)
            judge = (norm, judge_mistakes)
            rem, rpm... = xs[10:12]
            red_mistakes = (rem, rpm)
            red = (red_strategy, red_mistakes)
            bem, bpm... = xs[13:15]
            blue_mistakes = (bem, bpm)
            blue = (blue_strategy, blue_mistakes)
            mutant_rule = SVector{4,Bool}((false, false, false, false))
            red_mutant = (mutant_rule, red_mistakes)
            blue_mutant = (mutant_rule, blue_mistakes)
            R★, B★, RM★, BM★ = stationary_reputations(judge, red, blue, red_mutant, blue_mutant, proportion_incumbents_red)
            _, payoff_blue_mutant = mutant_payoffs(red, blue, red_mutant, blue_mutant, R★, B★, RM★, BM★, proportion_incumbents_red, utilities)
            return payoff_blue_mutant
        end
    end
end

function summarise_solution(model, values)
    pR, utilities, mistakes = values[1], values[2:5], values[6:15]
    ov = objective_value(model)
    pr = value(pR)
    ut = value.(utilities)
    ms = value.(mistakes)
    r(v) = round(v; sigdigits=3)
    println("### Solution Summary ###")
    println("Objective value: $ov")
    println("")
    println("Judge:")
    println("- Norm: $norm")
    @printf "- ε: %.2f\n" r(ms[1])
    @printf "- α: %.2f,\n     %.2f,\n     %.2f\n" r.(ms[2:4])...
    println("")
    println("Majority:")
    println("- Strategy: $red_strategy")
    @printf "- ε: %.2f\n" r(ms[5])
    @printf "- α: %.2f,\n     %.2f\n" r.(ms[6:7])...
    println("")
    println("Minority:")
    println("- Strategy: $blue_strategy")
    @printf "- ε: %.2f\n" r(ms[8])
    @printf "- α: %.2f,\n     %.2f\n" r.(ms[9:10])...
    println("")
    println("External factors:")
    println("- Majority proportion: $(r(pr))")
    println("- Benefit/cost:")
    println("  - Benefit maj: $(r(ut[1]))")
    println("  - Benefit min: $(r(ut[2]))")
    println("  - Cost maj: $(r(ut[3]))")
    println("  - Cost min: $(r(ut[4]))")
end