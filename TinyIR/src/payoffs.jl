# Donation probabilities derived from reputations
# function p_donates(agent, ingroup_reputation, outgroup_reputation, ingroup_proportion)
#     outgroup_proportion = 1 - ingroup_proportion
#     prob_to_donate =
#         ingroup_proportion * ingroup_reputation * evaluate(agent, (true, true)) +
#         outgroup_proportion * outgroup_reputation * evaluate(agent, (false, true)) +
#         ingroup_proportion * (1 - ingroup_reputation) * evaluate(agent, (true, false)) +
#         outgroup_proportion * (1 - outgroup_reputation) * evaluate(agent, (false, false))
#     return prob_to_donate
# end

function p_donates(a, iR, oR, pI)
    outgroup_proportion = 1 - pI
    prob_to_donate =
        pI * iR * evaluate(a, (true, true)) +
        outgroup_proportion * oR * evaluate(a, (false, true)) +
        pI * (1 - iR) * evaluate(a, (true, false)) +
        outgroup_proportion * (1 - oR) * evaluate(a, (false, false))
    return prob_to_donate
end

function p_receives(ingroup_agent, outgroup_agent, my_reputation, ingroup_proportion)
    ia_rule, ia_mistakes = ingroup_agent
    oa_rule, oa_mistakes = outgroup_agent
    outgroup_proportion = 1 - ingroup_proportion
    prob_to_receive =
        ingroup_proportion * my_reputation * evaluate(ia_rule, ia_mistakes, (true, true)) +
        outgroup_proportion * my_reputation * evaluate(oa_rule, oa_mistakes, (false, true)) +
        ingroup_proportion * (1 - my_reputation) * evaluate(ia_rule, ia_mistakes, (true, false)) +
        outgroup_proportion * (1 - my_reputation) * evaluate(oa_rule, oa_mistakes, (false, false))
    return prob_to_receive
end

# Payoffs derived from donation probabilities
function incumbent_payoffs(red, blue, red_reputation, blue_reputation, proportion_incumbents_red, utilities)
    benefit_red, benefit_blue, cost_red, cost_blue = utilities
    pR = proportion_incumbents_red
    R★ = red_reputation
    B★ = blue_reputation
    red_payoff =
        benefit_red * p_receives(red, blue, R★, pR) -
        cost_red * p_donates(red, R★, B★, pR)
    blue_payoff =
        benefit_blue * p_receives(blue, red, B★, 1 - pR) -
        cost_blue * p_donates(blue, B★, R★, 1 - pR)
    return (red_payoff, blue_payoff)
end

function mutant_payoffs(red, blue, red_mutant, blue_mutant, R★, B★, RM★, BM★, pR, utilities)
    benefit_red, benefit_blue, cost_red, cost_blue = utilities
    red_mutant_payoff =
        benefit_red * p_receives(red, blue, RM★, pR) -
        cost_red * p_donates(red_mutant, R★, B★, pR)
    blue_mutant_payoff =
        benefit_blue * p_receives(blue, red, BM★, 1 - pR) -
        cost_blue * p_donates(blue_mutant, B★, R★, 1 - pR)
    return (red_mutant_payoff, blue_mutant_payoff)
end

function payoffs(red, blue, red_mutant, blue_mutant, R★, B★, RM★, BM★, pR, utilities)
    return (
        incumbent_payoffs(red, blue, R★, B★, pR, utilities)...,
        mutant_payoffs(red, blue, red_mutant, blue_mutant, R★, B★, RM★, BM★, pR, utilities)...,
    )
end

# convenience payoffs functions
function payoffs(judge, red, blue, red_mutant, blue_mutant, proportion_incumbents_red, utilities)
    R★, B★, RM★, BM★ = stationary_reputations(judge, red, blue, red_mutant, blue_mutant, proportion_incumbents_red)
    return payoffs(red, blue, red_mutant, blue_mutant, R★, B★, RM★, BM★, proportion_incumbents_red, utilities)
end

# ESS
function is_ESS(judge, red, blue, proportion_incumbents_red, utilities)
    red_rule, red_mistakes = red
    blue_rule, blue_mistakes = blue
    for i in 0:15
        mutant_rule = SVector{4,Bool}(i >> shift & 1 != 0 for shift in 0:3)
        # mutant_rule = SVector{4,Bool}(digits(i, base=2, pad=4))
        # @show mutant_rule mr
        mutant_rule == red_rule == blue_rule && continue # Skip if strategies are all identical
        red_mutant = (mutant_rule, red_mistakes)
        blue_mutant = (mutant_rule, blue_mistakes)
        payoff_red, payoff_blue, payoff_red_mutant, payoff_blue_mutant =
            payoffs(judge, red, blue, red_mutant, blue_mutant, proportion_incumbents_red, utilities)
        if mutant_rule != red_rule
            (payoff_red >= payoff_red_mutant) || return false
        end
        if mutant_rule != blue_rule
            payoff_blue >= payoff_blue_mutant || return false
        end
    end
    return true
end

function invader(judge, red, blue, proportion_incumbents_red, utilities)
    red_rule, red_mistakes = red
    blue_rule, blue_mistakes = blue
    for i in 0:15
        mutant_rule = SVector{4,Bool}(digits(i, base=2, pad=4))
        mutant_rule == red_rule == blue_rule && continue # Skip if strategies are all identical
        red_mutant = (mutant_rule, red_mistakes)
        blue_mutant = (mutant_rule, blue_mistakes)
        payoff_red, payoff_blue, payoff_red_mutant, payoff_blue_mutant =
            payoffs(judge, red, blue, red_mutant, blue_mutant, proportion_incumbents_red, utilities)
        if mutant_rule != red_rule
            (payoff_red >= payoff_red_mutant) || return mutant_rule
        end
        if mutant_rule != blue_rule
            payoff_blue >= payoff_blue_mutant || return mutant_rule
        end
    end
    return nothing
end