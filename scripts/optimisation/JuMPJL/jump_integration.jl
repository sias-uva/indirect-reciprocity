# Objective function
function unpack_x(xs)
    prop_red = xs[1]
    utilities = xs[2:5]
    jem, jpm... = xs[6:9]
    rem, rpm... = xs[10:12]
    bem, bpm... = xs[13:15]
    return prop_red, utilities, jem, SA[jpm...], rem, SA[rpm...], bem, SA[bpm...]
end


function avg_payoffs(xs...; p)
    prop_red, utilities, jem, jpm, rem, rpm, bem, bpm = unpack_x(xs)
    judge = Agent(p.norm, jem, jpm)
    red = Agent(p.red_strategy, rem, rpm)
    blue = Agent(p.blue_strategy, bem, bpm)
    R★, B★ = stationary_incumbent_reputations(judge, red, blue, prop_red)
    avg = sum(incumbent_payoffs(red, blue, R★, B★, prop_red, utilities)) / 2
    return avg
end

# Helper function
function both_payoffs(xs...; p)
    prop_red, utilities, jem, jpm, rem, rpm, bem, bpm = unpack_x(xs)
    judge = Agent(p.norm, jem, jpm)
    red = Agent(p.red_strategy, rem, rpm)
    blue = Agent(p.blue_strategy, bem, bpm)
    R★, B★ = stationary_incumbent_reputations(judge, red, blue, prop_red)
    return incumbent_payoffs(red, blue, R★, B★, prop_red, utilities)
end

# Unused constraint
function ESS_constraint(xs...; p)
    prop_red, utilities, jem, jpm, rem, rpm, bem, bpm = unpack_x(xs)
    judge = Agent(p.norm, jem, jpm)
    red = Agent(p.red_strategy, rem, rpm)
    blue = Agent(p.blue_strategy, bem, bpm)
    if is_ESS(judge, red, blue, prop_red, utilities)
        return true
    else
        invader(judge, red, blue, prop_red, utilities)
    end
end

function payoff_incumbent_i(xs...; p, i_out)
    prop_red, utilities, jem, jpm, rem, rpm, bem, bpm = unpack_x(xs)
    judge = Agent(p.norm, jem, jpm)
    red = Agent(p.red_strategy, rem, rpm)
    blue = Agent(p.blue_strategy, bem, bpm)
    R★, B★ = stationary_incumbent_reputations(judge, red, blue, prop_red)
    ip = incumbent_payoffs(red, blue, R★, B★, prop_red, utilities)
    return ip[i_out]
end

function mutant_payoff_i(xs...; p, mutant_rule, out_i)
    prop_red, utilities, jem, jpm, rem, rpm, bem, bpm = unpack_x(xs)
    judge = Agent(p.norm, jem, jpm)
    red = Agent(p.red_strategy, rem, rpm)
    blue = Agent(p.blue_strategy, bem, bpm)
    R★, B★ = stationary_incumbent_reputations(judge, red, blue, prop_red)
    
    red_mutant = Agent(mutant_rule, rem, rpm)
    blue_mutant = Agent(mutant_rule, bem, bpm)
    RM★, BM★ = stationary_mutant_reputations(judge, red_mutant, blue_mutant, R★, B★, prop_red)
    payoff_mutants = mutant_payoffs(red, blue, red_mutant, blue_mutant, R★, B★, RM★, BM★, prop_red, utilities)
    return payoff_mutants[out_i]
end


function summarise_solution(model, values; p)
    prop_red, utilities, mistakes = values[1], values[2:5], values[6:15]
    ov = objective_value(model)
    pr = value(prop_red)
    ut = value.(utilities)
    ms = value.(mistakes)
    r(v) = round(v; sigdigits=3)
    println("### Solution Summary ###")
    println("Objective value: $ov")
    println("")
    println("Judge:")
    println("- Norm: $(p.norm)")
    @printf "- ε: %.2f\n" r(ms[1])
    @printf "- α: %.2f,\n     %.2f,\n     %.2f\n" r.(ms[2:4])...
    println("")
    println("Majority:")
    println("- Strategy: $(p.red_strategy)")
    @printf "- ε: %.2f\n" r(ms[5])
    @printf "- α: %.2f,\n     %.2f\n" r.(ms[6:7])...
    println("")
    println("Minority:")
    println("- Strategy: $(p.blue_strategy)")
    @printf "- ε: %.2f\n" r(ms[8])
    @printf "- α: %.2f,\n     %.2f\n" r.(ms[9:10])...
    println("")
    println("External factors:")
    println("- Majority proportion: $(r(pr))")
    println("- Benefit/cost:")
    println("  - Benefit maj: $(r(ut[1]))")
    println("  - Benefit minority: $(r(ut[2]))")
    println("  - Cost maj: $(r(ut[3]))")
    println("  - Cost minority: $(r(ut[4]))")
end