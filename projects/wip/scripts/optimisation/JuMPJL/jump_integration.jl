# Objective function
function unpack_x(xs)
    prop_majority = xs[1]
    utilities = xs[2:5]
    judge_em, judge_pm... = xs[6:9]
    maj_em, maj_pm... = xs[10:12]
    min_em, min_pm... = xs[13:15]
    return prop_majority,
    utilities, judge_em, SA[judge_pm...], maj_em, SA[maj_pm...], min_em,
    SA[min_pm...]
end

function avg_payoffs(xs...; p)
    prop_majority, utilities, judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm = unpack_x(
        xs
    )
    judge = Agent(p.norm, judge_em, judge_pm)
    majority = Agent(p.majority_strategy, maj_em, maj_pm)
    minority = Agent(p.minority_strategy, min_em, min_pm)
    R★, B★ = stationary_incumbent_reputations(judge, majority, minority, prop_majority)
    avg = sum(incumbent_payoffs(majority, minority, R★, B★, prop_majority, utilities)) / 2
    return avg
end

# Helper function
function both_payoffs(xs...; p)
    prop_majority, utilities, judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm = unpack_x(
        xs
    )
    judge = Agent(p.norm, judge_em, judge_pm)
    majority = Agent(p.majority_strategy, maj_em, maj_pm)
    minority = Agent(p.minority_strategy, min_em, min_pm)
    R★, B★ = stationary_incumbent_reputations(judge, majority, minority, prop_majority)
    return incumbent_payoffs(majority, minority, R★, B★, prop_majority, utilities)
end

# Unused constraint
function ESS_constraint(xs...; p)
    prop_majority, utilities, judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm = unpack_x(
        xs
    )
    judge = Agent(p.norm, judge_em, judge_pm)
    majority = Agent(p.majority_strategy, maj_em, maj_pm)
    minority = Agent(p.minority_strategy, min_em, min_pm)
    if is_ESS(judge, majority, minority, prop_majority, utilities)
        return true
    else
        invader(judge, majority, minority, prop_majority, utilities)
    end
end

function payoff_incumbent_i(xs...; p, i_out)
    prop_majority, utilities, judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm = unpack_x(
        xs
    )
    judge = Agent(p.norm, judge_em, judge_pm)
    majority = Agent(p.majority_strategy, maj_em, maj_pm)
    minority = Agent(p.minority_strategy, min_em, min_pm)
    R★, B★ = stationary_incumbent_reputations(judge, majority, minority, prop_majority)
    ip = incumbent_payoffs(majority, minority, R★, B★, prop_majority, utilities)
    return ip[i_out]
end

function mutant_payoff_i(xs...; p, mutant_rule, out_i)
    prop_majority, utilities, judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm = unpack_x(
        xs
    )
    judge = Agent(p.norm, judge_em, judge_pm)
    majority = Agent(p.majority_strategy, maj_em, maj_pm)
    minority = Agent(p.minority_strategy, min_em, min_pm)
    R★, B★ = stationary_incumbent_reputations(judge, majority, minority, prop_majority)

    majority_mutant = Agent(mutant_rule, maj_em, maj_pm)
    minority_mutant = Agent(mutant_rule, min_em, min_pm)
    RM★, BM★ = stationary_mutant_reputations(
        judge, majority_mutant, minority_mutant, R★, B★, prop_majority
    )
    payoff_mutants = mutant_payoffs(
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
    return payoff_mutants[out_i]
end

function summarise_solution(model, values; p)
    prop_majority, utilities, mistakes = values[1], values[2:5], values[6:15]
    ov = objective_value(model)
    pr = value(prop_majority)
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
    println("- Strategy: $(p.majority_strategy)")
    @printf "- ε: %.2f\n" r(ms[5])
    @printf "- α: %.2f,\n     %.2f\n" r.(ms[6:7])...
    println("")
    println("Minority:")
    println("- Strategy: $(p.minority_strategy)")
    @printf "- ε: %.2f\n" r(ms[8])
    @printf "- α: %.2f,\n     %.2f\n" r.(ms[9:10])...
    println("")
    println("External factors:")
    println("- Majority proportion: $(r(pr))")
    println("- Benefit/cost:")
    println("  - Benefit maj: $(r(ut[1]))")
    println("  - Benefit minority: $(r(ut[2]))")
    println("  - Cost maj: $(r(ut[3]))")
    return println("  - Cost minority: $(r(ut[4]))")
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
    payoff_majority = (args...) -> payoff_incumbent_i(args...; p=p, i_out=1)
    payoff_minority = (args...) -> payoff_incumbent_i(args...; p=p, i_out=2)
    register(model, :payoff_majority, 15, payoff_majority; autodiff=true)
    register(model, :payoff_minority, 15, payoff_minority; autodiff=true)

    # Add decision variables and their domains
    @variable(model, 0.5 <= prop_maj <= 0.9) # Size of the majority

    @variable(model, domain_eps <= benefits[1:2] <= benefit_max) # Benefits
    @variable(model, cost_min <= costs[1:2] <= Inf) # Costs

    @variable(model, domain_eps <= judge_em <= em_bound) # Rates of execution mistakes of judge
    @variable(model, domain_eps <= majority_em <= em_bound) # Rates of execution mistakes of majority agents
    @variable(model, domain_eps <= minority_em <= em_bound) # Rates of execution mistakes of minority agents

    @variable(model, domain_eps <= judge_pm[1:3] <= pm_bound) # Rates of perception mistakes of judge
    @variable(model, domain_eps <= majority_pm[1:2] <= pm_bound) # Rates of perception mistakes of agents
    @variable(model, domain_eps <= minority_pm[1:2] <= pm_bound) # Rates of perception mistakes of agents

    vars = (
        prop_maj,
        benefits...,
        costs...,
        judge_em,
        judge_pm...,
        majority_em,
        majority_pm...,
        minority_em,
        minority_pm...,
    ) # Put them all in a single variable

    # Add our non-linear objective function
    @NLobjective(model, Min, -payoff_majority(vars...) - payoff_minority(vars...)) # Maximise average payoffs...

    # ...subject to the constraint that the incumbent payoffs are larger or equal to
    # any possible mutant.
    # println("Registering constraints 0 to 15")
    majority_i = @evalpoly(2, majority_strategy...)
    minority_i = @evalpoly(2, minority_strategy...)
    # @show majority_i minority_i
    for i in 0:15
        mutant_rule = Strategy(digits(i; base=2, pad=4))
        majority_name = Symbol(string("payoff_majority_mutant_", i))
        minority_name = Symbol(string("payoff_minority_mutant_", i))
        register(
            model,
            majority_name,
            15,
            (args...) -> mutant_payoff_i(args...; p, mutant_rule, out_i=1);
            autodiff=true,
        )
        register(
            model,
            minority_name,
            15,
            (args...) -> mutant_payoff_i(args...; p, mutant_rule, out_i=2);
            autodiff=true,
        )
        if i != majority_i
            @eval @NLconstraint(
                $model,
                payoff_majority($(vars)...) >= $domain_eps + $(majority_name)($(vars)...)
            )
        end
        if i != minority_i
            @eval @NLconstraint(
                $model,
                payoff_minority($(vars)...) >= $domain_eps + $(minority_name)($(vars)...)
            )
        end
    end

    # Constraint on fairness
    cf1 = @NLconstraint(
        model, payoff_majority(vars...) >= fairness_bound * payoff_minority(vars...)
    )
    cf2 = @NLconstraint(
        model, payoff_minority(vars...) >= fairness_bound * payoff_majority(vars...)
    )

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
