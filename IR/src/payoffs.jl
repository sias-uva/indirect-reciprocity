# Donation probabilities
function p_donates(agent, ingroup_rep, outgroup_rep, ingroup_prop)
    return lerp(
        SA[
            lerp(SA[agent(0, 0), agent(0, 1)], outgroup_rep),
            lerp(SA[agent(1, 0), agent(1, 1)], ingroup_rep),
        ],
        ingroup_prop,
    )
end

function p_receives(ingroup_agent, outgroup_agent, my_rep, ingroup_prop)
    return lerp(
        SA[
            lerp(SA[outgroup_agent(0, 0), outgroup_agent(0, 1)], my_rep),
            lerp(SA[ingroup_agent(1, 0), ingroup_agent(1, 1)], my_rep),
        ],
        ingroup_prop,
    )
end

# Payoffs derived from donation probabilities
function incumbent_payoffs(red, blue, red_rep, blue_rep, prop_red, utilities)
    benefit_red, benefit_blue, cost_red, cost_blue = utilities
    red_payoff =
        benefit_red * p_receives(red, blue, red_rep, prop_red) -
        cost_red * p_donates(red, red_rep, blue_rep, prop_red)
    blue_payoff =
        benefit_blue * p_receives(blue, red, blue_rep, 1 - prop_red) -
        cost_blue * p_donates(blue, blue_rep, red_rep, 1 - prop_red)
    return SA[red_payoff, blue_payoff]
end

function mutant_payoffs(
    red,
    blue,
    red_mutant,
    blue_mutant,
    red_rep,
    blue_rep,
    red_mutant_rep,
    blue_mutant_rep,
    prop_red,
    utilities,
)
    benefit_red, benefit_blue, cost_red, cost_blue = utilities
    payoff_red_mutant =
        benefit_red * p_receives(red, blue, red_mutant_rep, prop_red) -
        cost_red * p_donates(red_mutant, red_rep, blue_rep, prop_red)
    payoff_blue_mutant =
        benefit_blue *
        p_receives(blue, red, blue_mutant_rep, 1 - prop_red) -
        cost_blue *
        p_donates(blue_mutant, blue_rep, red_rep, 1 - prop_red)
    return SA[payoff_red_mutant, payoff_blue_mutant]
end

function payoffs(
    red,
    blue,
    red_mutant,
    blue_mutant,
    red_rep,
    blue_rep,
    red_mutant_rep,
    blue_mutant_rep,
    prop_red,
    utilities,
)
    ip = incumbent_payoffs(red, blue, red_rep, blue_rep, prop_red, utilities)
    mp = mutant_payoffs(
        red,
        blue,
        red_mutant,
        blue_mutant,
        red_rep,
        blue_rep,
        red_mutant_rep,
        blue_mutant_rep,
        prop_red,
        utilities,
    )
    return [ip mp]
end

function payoffs(judge, red, blue, red_mutant, blue_mutant, prop_red, utilities)
    reputations = stationary_reputations(
        judge, red, blue, red_mutant, blue_mutant, prop_red
    )
    return payoffs(red, blue, red_mutant, blue_mutant, reputations..., prop_red, utilities)
end

function is_ESS(judge, red::Player, blue::Player, prop_red, utilities)
    for i in 0:15
        mutant_strategy = SMatrix{2,2,Bool}(i >> shift & 1 != 0 for shift in 0:3)
        mutant_strategy == red.rule == blue.rule && continue # Skip if strategies are all identical
        red_mutant = Agent(mutant_strategy, red.ε, red.α)
        blue_mutant = Agent(mutant_strategy, blue.ε, blue.α)
        payoff_red, payoff_blue, payoff_red_mutant, payoff_blue_mutant = payoffs(
            judge, red, blue, red_mutant, blue_mutant, prop_red, utilities
        )
        if mutant_strategy != red.rule
            (payoff_red >= payoff_red_mutant) || return false
        end
        if mutant_strategy != blue.rule
            payoff_blue >= payoff_blue_mutant || return false
        end
    end
    return true
end

function invader(judge, red::Player, blue::Player, prop_red, utilities)
    for i in 0:15
        mutant_strategy = SMatrix{2,2,Bool}(i >> shift & 1 != 0 for shift in 0:3)
        mutant_strategy == red.rule == blue.rule && continue # Skip if strategies are all identical
        red_mutant = Agent(mutant_strategy, red.ε, red.α)
        blue_mutant = Agent(mutant_strategy, blue.ε, blue.α)
        payoff_red, payoff_blue, payoff_red_mutant, payoff_blue_mutant = payoffs(
            judge, red, blue, red_mutant, blue_mutant, prop_red, utilities
        )
        if mutant_strategy != red.rule
            (payoff_red >= payoff_red_mutant) || return ("red", red, red_mutant)
        end
        if mutant_strategy != blue.rule
            payoff_blue >= payoff_blue_mutant || return ("blue", blue, blue_mutant)
        end
    end
    return nothing
end