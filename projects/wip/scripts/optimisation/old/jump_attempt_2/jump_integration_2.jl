# Objective function
function unpack_x(xs)
    prop_majority = xs[1]
    utilities = xs[2:5]
    judge_em, judge_pm... = xs[6:9]
    maj_em, maj_pm... = xs[10:12]
    min_em, min_pm... = xs[13:15]
    reputations = xs[16:17]
    return prop_majority,
    utilities, judge_em, SA[judge_pm...], maj_em, SA[maj_pm...], min_em, SA[min_pm...],
    reputations
end

function avg_payoffs(xs...)
    prop_majority, utilities, judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm, reputations = unpack_x(
        xs
    )
    majority = Agent(majority_strategy, maj_em, maj_pm)
    minority = Agent(minority_strategy, min_em, min_pm)
    R★, B★ = reputations
    avg = sum(incumbent_payoffs(majority, minority, R★, B★, prop_majority, utilities)) / 2
    return avg
end

# Helper function
function both_payoffs(xs...)
    prop_majority, utilities, judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm, reputations = unpack_x(
        xs
    )
    majority = Agent(majority_strategy, maj_em, maj_pm)
    minority = Agent(minority_strategy, min_em, min_pm)
    R★, B★ = reputations
    return incumbent_payoffs(majority, minority, R★, B★, prop_majority, utilities)
end

# Constraint
function ESS_constraint(xs...)
    prop_majority, utilities, judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm, reputations = unpack_x(
        xs
    )
    judge = Agent(norm, judge_em, judge_pm)
    majority = Agent(majority_strategy, maj_em, maj_pm)
    minority = Agent(minority_strategy, min_em, min_pm)
    @show minority_strategy
    if is_ESS(judge, majority, minority, prop_majority, utilities)
        return true
    else
        invader(judge, majority, minority, prop_majority, utilities)
    end
end

function payoff_majority(xs...)
    prop_majority, utilities, judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm, reputations = unpack_x(
        xs
    )
    judge = Agent(norm, judge_em, judge_pm)
    majority = Agent(majority_strategy, maj_em, maj_pm)
    minority = Agent(minority_strategy, min_em, min_pm)
    R★, B★ = reputations
    payoff_majority, _ = incumbent_payoffs(
        majority, minority, R★, B★, prop_majority, utilities
    )
    return payoff_majority
end

function payoff_minority(xs...)
    prop_majority, utilities, judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm, reputations = unpack_x(
        xs
    )
    judge = Agent(norm, judge_em, judge_pm)
    majority = Agent(majority_strategy, maj_em, maj_pm)
    minority = Agent(minority_strategy, min_em, min_pm)
    R★, B★ = reputations
    _, payoff_minority = incumbent_payoffs(
        majority, minority, R★, B★, prop_majority, utilities
    )
    return payoff_minority
end

function stationary_reps(xs...)
    prop_majority, utilities, judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm, reputations = unpack_x(
        xs
    )
    judge = Agent(norm, judge_em, judge_pm)
    majority = Agent(majority_strategy, maj_em, maj_pm)
    minority = Agent(minority_strategy, min_em, min_pm)
    R★, B★ = reputations
    return stationary_mutant_reputations(judge, majority, minority, R★, B★, prop_majority)[1]
end

function stationary_reps2(xs...)
    prop_majority, utilities, judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm, reputations = unpack_x(
        xs
    )
    judge = Agent(norm, judge_em, judge_pm)
    majority = Agent(majority_strategy, maj_em, maj_pm)
    minority = Agent(minority_strategy, min_em, min_pm)
    R★, B★ = reputations
    return stationary_mutant_reputations(judge, majority, minority, R★, B★, prop_majority)[2]
end

for i in 0:15
    mr = Strategy(digits(i; base=2, pad=4))
    majority_name = Symbol(string("payoff_majority_mutant_", i))
    minority_name = Symbol(string("payoff_minority_mutant_", i))
    mutant_rule = SVector{4,Bool}((false, false, false, false))
    @eval begin
        function $(majority_name)(xs...)
            prop_majority, utilities, judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm, reputations = unpack_x(
                xs
            )
            judge = Agent(norm, judge_em, judge_pm)
            majority = Agent(majority_strategy, maj_em, maj_pm)
            minority = Agent(minority_strategy, min_em, min_pm)
            R★, B★ = reputations
            mutant_rule = $(mr)
            majority_mutant = Agent(mutant_rule, maj_em, maj_pm)
            minority_mutant = Agent(mutant_rule, min_em, min_pm)
            RM★, BM★ = stationary_mutant_reputations(
                judge, majority, minority, R★, B★, prop_majority
            )
            payoff_majority_mutant, _ = mutant_payoffs(
                majority,
                minority,
                majority_mutant,
                minority_mutant,
                R★,
                B★,
                RM★,
                BM★,
                prop_majority,
                utilities,
            )
            return payoff_majority_mutant
        end
        function $(minority_name)(xs...)
            prop_majority, utilities, judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm, reputations = unpack_x(
                xs
            )
            judge = Agent(norm, judge_em, judge_pm)
            majority = Agent(majority_strategy, maj_em, maj_pm)
            minority = Agent(minority_strategy, min_em, min_pm)
            R★, B★ = reputations
            mutant_rule = $(mr)
            majority_mutant = Agent(mutant_rule, maj_em, maj_pm)
            minority_mutant = Agent(mutant_rule, min_em, min_pm)
            RM★, BM★ = stationary_mutant_reputations(
                judge, majority, minority, R★, B★, prop_majority
            )
            _, payoff_minority_mutant = mutant_payoffs(
                majority,
                minority,
                majority_mutant,
                minority_mutant,
                R★,
                B★,
                RM★,
                BM★,
                prop_majority,
                utilities,
            )
            return payoff_minority_mutant
        end
    end
end

function summarise_solution(model, values)
    prop_maj, utilities, mistakes = values[1], values[2:5], values[6:15]
    ov = objective_value(model)
    pr = value(prop_maj)
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
    println("- Strategy: $majority_strategy")
    @printf "- ε: %.2f\n" r(ms[5])
    @printf "- α: %.2f,\n     %.2f\n" r.(ms[6:7])...
    println("")
    println("Minority:")
    println("- Strategy: $minority_strategy")
    @printf "- ε: %.2f\n" r(ms[8])
    @printf "- α: %.2f,\n     %.2f\n" r.(ms[9:10])...
    println("")
    println("External factors:")
    println("- Majority proportion: $(r(pr))")
    println("- Benefit/cost:")
    println("  - Benefit maj: $(r(ut[1]))")
    println("  - Benefit min: $(r(ut[2]))")
    println("  - Cost maj: $(r(ut[3]))")
    return println("  - Cost min: $(r(ut[4]))")
end
