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
    RM★, BM★ =
        stationary_mutant_reputations(judge, red_mutant, blue_mutant, R★, B★, prop_red)
    payoff_mutants = mutant_payoffs(
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
    return payoff_mutants[out_i]
end


function summarise_solution(model, values; p)
    prop_red, utilities, mistakes = values[1], values[2:5], values[6:15]
    ov = objective_value(model)
    pr = value(prop_red)
    ut = value.(utilities)
    ms = value.(mistakes)
    r(v) = round(v; sigdigits = 3)
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

function cooperation_fairness(vars)
    r, b = both_payoffs(value.(vars)...; p)
    fairness = r > b ? b / r : r / b
    cooperation = (r + b) / 2
    return cooperation, fairness
end


function create_model(fairness_bound, p)
    # Initialise model with chosen solver
    model = Model(Ipopt.Optimizer)

    # Register our custom objective function and auxiliary functions useful for constraints
    payoff_red = (args...) -> payoff_incumbent_i(args...; p = p, i_out = 1)
    payoff_blue = (args...) -> payoff_incumbent_i(args...; p = p, i_out = 2)
    register(model, :payoff_red, 15, payoff_red; autodiff = true)
    register(model, :payoff_blue, 15, payoff_blue; autodiff = true)

    # Add decision variables and their domains
    @variable(model, 0.5 <= pR <= 0.9) # Size of the majority

    @variable(model, domain_eps <= benefits[1:2] <= benefit_max) # Benefits
    @variable(model, cost_min <= costs[1:2] <= Inf) # Costs

    @variable(model, domain_eps <= judge_em <= em_bound) # Rates of execution mistakes of judge
    @variable(model, domain_eps <= red_em <= em_bound) # Rates of execution mistakes of red agents
    @variable(model, domain_eps <= blue_em <= em_bound) # Rates of execution mistakes of blue agents

    @variable(model, domain_eps <= judge_pm[1:3] <= pm_bound) # Rates of perception mistakes of judge
    @variable(model, domain_eps <= red_pm[1:2] <= pm_bound) # Rates of perception mistakes of agents
    @variable(model, domain_eps <= blue_pm[1:2] <= pm_bound) # Rates of perception mistakes of agents

    vars = (
        pR,
        benefits...,
        costs...,
        judge_em,
        judge_pm...,
        red_em,
        red_pm...,
        blue_em,
        blue_pm...,
    ) # Put them all in a single variable

    # Add our non-linear objective function
    @NLobjective(model, Min, -payoff_red(vars...) - payoff_blue(vars...)) # Maximise average payoffs...

    # ...subject to the constraint that the incumbent payoffs are larger or equal to
    # any possible mutant.
    # println("Registering constraints 0 to 15")
    red_i = @evalpoly(2, red_strategy...)
    blue_i = @evalpoly(2, blue_strategy...)
    # @show red_i blue_i
    for i = 0:15
        mutant_rule = Strategy(digits(i, base = 2, pad = 4))
        red_name = Symbol(string("payoff_red_mutant_", i))
        blue_name = Symbol(string("payoff_blue_mutant_", i))
        register(
            model,
            red_name,
            15,
            (args...) -> mutant_payoff_i(args...; p, mutant_rule, out_i = 1);
            autodiff = true,
        )
        register(
            model,
            blue_name,
            15,
            (args...) -> mutant_payoff_i(args...; p, mutant_rule, out_i = 2);
            autodiff = true,
        )
        if i != red_i
            @eval @NLconstraint(
                $model,
                payoff_red($(vars)...) >= $domain_eps + $(red_name)($(vars)...)
            )
        end
        if i != blue_i
            @eval @NLconstraint(
                $model,
                payoff_blue($(vars)...) >= $domain_eps + $(blue_name)($(vars)...)
            )
        end
    end

    # Constraint on fairness
    cf1 = @NLconstraint(model, payoff_red(vars...) >= fairness_bound * payoff_blue(vars...))
    cf2 = @NLconstraint(model, payoff_blue(vars...) >= fairness_bound * payoff_red(vars...))

    # Constrain the utilities (although some are superfluous)
    begin
        # Benefits must be greater than costs
        @constraint(model, [i = 1:2], costs[i] <= benefits[i])

        # @NLconstraint(model)
        # Sum of benefits can't exceed predetermined maximum
        @constraint(model, benefits[1] + benefits[2] <= benefit_sum_max)
        # Sum of costs must be greater than predetermined minimum
        @constraint(model, costs[1] + costs[2] >= cost_sum_min)
        # Groups must have the same benefits...
        @constraint(model, benefits[1] == benefits[2])
        # ...and costs
        @constraint(model, costs[1] == costs[2])
        # set_optimizer_attribute(model, "max_iter", 4000) # The solver can take a LOT of iterations before it finds an optimal solution
    end
    return model
end
