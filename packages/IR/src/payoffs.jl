# Donation probabilities
function p_donates(agent, ingroup_rep, outgroup_rep, ingroup_prop)
    return lerp(
        SA[
            lerp(SA[agent(false, false), agent(false, true)], outgroup_rep),
            lerp(SA[agent(true, false), agent(true, true)], ingroup_rep),
        ],
        ingroup_prop,
    )
end

function p_receives(ingroup_agent, outgroup_agent, my_rep, ingroup_prop)
    return lerp(
        SA[
            lerp(SA[outgroup_agent(false, false), outgroup_agent(false, true)], my_rep),
            lerp(SA[ingroup_agent(true, false), ingroup_agent(true, true)], my_rep),
        ],
        ingroup_prop,
    )
end

# Payoffs derived from donation probabilities
function incumbent_payoffs(
    majority, minority, majority_rep, minority_rep, prop_majority, utilities
)
    benefit_majority, benefit_minority, cost_majority, cost_minority = utilities
    majority_payoff =
        benefit_majority * p_receives(majority, minority, majority_rep, prop_majority) -
        cost_majority * p_donates(majority, majority_rep, minority_rep, prop_majority)
    minority_payoff =
        benefit_minority * p_receives(minority, majority, minority_rep, 1 - prop_majority) -
        cost_minority * p_donates(minority, minority_rep, majority_rep, 1 - prop_majority)
    return SA[majority_payoff, minority_payoff]
end

function mutant_payoffs(
    majority,
    minority,
    majority_mutant,
    minority_mutant,
    majority_rep,
    minority_rep,
    majority_mutant_rep,
    minority_mutant_rep,
    prop_majority,
    utilities,
)
    benefit_majority, benefit_minority, cost_majority, cost_minority = utilities
    payoff_majority_mutant =
        benefit_majority *
        p_receives(majority, minority, majority_mutant_rep, prop_majority) -
        cost_majority *
        p_donates(majority_mutant, majority_rep, minority_rep, prop_majority)
    payoff_minority_mutant =
        benefit_minority *
        p_receives(minority, majority, minority_mutant_rep, 1 - prop_majority) -
        cost_minority *
        p_donates(minority_mutant, minority_rep, majority_rep, 1 - prop_majority)
    return SA[payoff_majority_mutant, payoff_minority_mutant]
end

function payoffs(
    majority,
    minority,
    majority_mutant,
    minority_mutant,
    majority_rep,
    minority_rep,
    majority_mutant_rep,
    minority_mutant_rep,
    prop_majority,
    utilities,
)
    ip = incumbent_payoffs(
        majority, minority, majority_rep, minority_rep, prop_majority, utilities
    )
    mp = mutant_payoffs(
        majority,
        minority,
        majority_mutant,
        minority_mutant,
        majority_rep,
        minority_rep,
        majority_mutant_rep,
        minority_mutant_rep,
        prop_majority,
        utilities,
    )
    return [ip mp]
end

function payoffs(
    judge, majority, minority, majority_mutant, minority_mutant, prop_majority, utilities
)
    reputations = stationary_reputations(
        judge, majority, minority, majority_mutant, minority_mutant, prop_majority
    )
    return payoffs(
        majority,
        minority,
        majority_mutant,
        minority_mutant,
        reputations...,
        prop_majority,
        utilities,
    )
end

"""
    is_ESS(judge, majority::Player, minority::Player, prop_majority, utilities)

Determines whether the system specified by arguments is an evolutionaily stable
state.
"""
function is_ESS(
    judge,
    majority::Player,
    minority::Player,
    prop_majority,
    utilities;
    mutant_filter=x -> true,
)
    for i in 0:15
        mutant_filter(i) || continue
        mutant_strategy = SMatrix{2,2,Bool}(i >> shift & 1 != 0 for shift in 0:3)
        mutant_strategy == majority.rule == minority.rule && continue # Skip if strategies are all identical
        majority_mutant = Agent(mutant_strategy, majority.ε, majority.α)
        minority_mutant = Agent(mutant_strategy, minority.ε, minority.α)
        payoff_majority, payoff_minority, payoff_majority_mutant, payoff_minority_mutant = payoffs(
            judge,
            majority,
            minority,
            majority_mutant,
            minority_mutant,
            prop_majority,
            utilities,
        )
        if mutant_strategy != majority.rule
            (payoff_majority >= payoff_majority_mutant) || return false
        end
        if mutant_strategy != minority.rule
            payoff_minority >= payoff_minority_mutant || return false
        end
    end
    return true
end

function invader(judge, majority::Player, minority::Player, prop_majority, utilities)
    for i in 0:15
        mutant_strategy = SMatrix{2,2,Bool}(i >> shift & 1 != 0 for shift in 0:3)
        mutant_strategy == majority.rule == minority.rule && continue # Skip if strategies are all identical
        majority_mutant = Agent(mutant_strategy, majority.ε, majority.α)
        minority_mutant = Agent(mutant_strategy, minority.ε, minority.α)
        payoff_majority, payoff_minority, payoff_majority_mutant, payoff_minority_mutant = payoffs(
            judge,
            majority,
            minority,
            majority_mutant,
            minority_mutant,
            prop_majority,
            utilities,
        )
        if mutant_strategy != majority.rule
            (payoff_majority >= payoff_majority_mutant) || return (
                "majority",
                majority.rule,
                majority_mutant.rule,
                payoff_majority,
                payoff_majority_mutant,
            )
        end
        if mutant_strategy != minority.rule
            payoff_minority >= payoff_minority_mutant || return (
                "minority",
                minority.rule,
                minority_mutant.rule,
                payoff_minority,
                payoff_minority_mutant,
            )
        end
    end
    return nothing
end
