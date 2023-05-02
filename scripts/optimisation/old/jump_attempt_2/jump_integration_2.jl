# Objective function
function unpack_x(xs)
    prop_red = xs[1]
    utilities = xs[2:5]
    jem, jpm... = xs[6:9]
    rem, rpm... = xs[10:12]
    bem, bpm... = xs[13:15]
    reputations = xs[16:17]
    return prop_red,
    utilities,
    jem,
    SA[jpm...],
    rem,
    SA[rpm...],
    bem,
    SA[bpm...],
    reputations
end

function avg_payoffs(xs...)
    prop_red, utilities, jem, jpm, rem, rpm, bem, bpm, reputations = unpack_x(xs)
    red = Agent(red_strategy, rem, rpm)
    blue = Agent(blue_strategy, bem, bpm)
    R★, B★ = reputations
    avg = sum(incumbent_payoffs(red, blue, R★, B★, prop_red, utilities)) / 2
    return avg
end

# Helper function
function both_payoffs(xs...)
    prop_red, utilities, jem, jpm, rem, rpm, bem, bpm, reputations = unpack_x(xs)
    red = Agent(red_strategy, rem, rpm)
    blue = Agent(blue_strategy, bem, bpm)
    R★, B★ = reputations
    return incumbent_payoffs(red, blue, R★, B★, prop_red, utilities)
end

# Constraint
function ESS_constraint(xs...)
    prop_red, utilities, jem, jpm, rem, rpm, bem, bpm, reputations = unpack_x(xs)
    judge = Agent(norm, jem, jpm)
    red = Agent(red_strategy, rem, rpm)
    blue = Agent(blue_strategy, bem, bpm)
    @show blue_strategy
    if is_ESS(judge, red, blue, prop_red, utilities)
        return true
    else
        invader(judge, red, blue, prop_red, utilities)
    end
end

function payoff_red(xs...)
    prop_red, utilities, jem, jpm, rem, rpm, bem, bpm, reputations = unpack_x(xs)
    judge = Agent(norm, jem, jpm)
    red = Agent(red_strategy, rem, rpm)
    blue = Agent(blue_strategy, bem, bpm)
    R★, B★ = reputations
    payoff_red, _ = incumbent_payoffs(red, blue, R★, B★, prop_red, utilities)
    return payoff_red
end

function payoff_blue(xs...)
    prop_red, utilities, jem, jpm, rem, rpm, bem, bpm, reputations = unpack_x(xs)
    judge = Agent(norm, jem, jpm)
    red = Agent(red_strategy, rem, rpm)
    blue = Agent(blue_strategy, bem, bpm)
    R★, B★ = reputations
    _, payoff_blue = incumbent_payoffs(red, blue, R★, B★, prop_red, utilities)
    return payoff_blue
end

function stationary_reps(xs...)
    prop_red, utilities, jem, jpm, rem, rpm, bem, bpm, reputations = unpack_x(xs)
    judge = Agent(norm, jem, jpm)
    red = Agent(red_strategy, rem, rpm)
    blue = Agent(blue_strategy, bem, bpm)
    R★, B★ = reputations
    return stationary_mutant_reputations(judge, red, blue, R★, B★, prop_red)[1]
end

function stationary_reps2(xs...)
    prop_red, utilities, jem, jpm, rem, rpm, bem, bpm, reputations = unpack_x(xs)
    judge = Agent(norm, jem, jpm)
    red = Agent(red_strategy, rem, rpm)
    blue = Agent(blue_strategy, bem, bpm)
    R★, B★ = reputations
    return stationary_mutant_reputations(judge, red, blue, R★, B★, prop_red)[2]
end

for i = 0:15
    mr = Strategy(digits(i, base = 2, pad = 4))
    red_name = Symbol(string("payoff_red_mutant_", i))
    blue_name = Symbol(string("payoff_blue_mutant_", i))
    mutant_rule = SVector{4,Bool}((false, false, false, false))
    @eval begin
        function $(red_name)(xs...)
            prop_red, utilities, jem, jpm, rem, rpm, bem, bpm, reputations = unpack_x(xs)
            judge = Agent(norm, jem, jpm)
            red = Agent(red_strategy, rem, rpm)
            blue = Agent(blue_strategy, bem, bpm)
            R★, B★ = reputations
            mutant_rule = $(mr)
            red_mutant = Agent(mutant_rule, rem, rpm)
            blue_mutant = Agent(mutant_rule, bem, bpm)
            RM★, BM★ = stationary_mutant_reputations(judge, red, blue, R★, B★, prop_red)
            payoff_red_mutant, _ = mutant_payoffs(
                red,
                blue,
                red_mutant,
                blue_mutant,
                R★,
                B★,
                RM★,
                BM★,
                prop_red,
                utilities,
            )
            return payoff_red_mutant
        end
        function $(blue_name)(xs...)
            prop_red, utilities, jem, jpm, rem, rpm, bem, bpm, reputations = unpack_x(xs)
            judge = Agent(norm, jem, jpm)
            red = Agent(red_strategy, rem, rpm)
            blue = Agent(blue_strategy, bem, bpm)
            R★, B★ = reputations
            mutant_rule = $(mr)
            red_mutant = Agent(mutant_rule, rem, rpm)
            blue_mutant = Agent(mutant_rule, bem, bpm)
            RM★, BM★ = stationary_mutant_reputations(judge, red, blue, R★, B★, prop_red)
            _, payoff_blue_mutant = mutant_payoffs(
                red,
                blue,
                red_mutant,
                blue_mutant,
                R★,
                B★,
                RM★,
                BM★,
                prop_red,
                utilities,
            )
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
    r(v) = round(v; sigdigits = 3)
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
