# Reputations derived from social rules
# where `agent` unpacks to `(agent.rule, (agent.em, agent.pm))`

# function stationary_incumbent_reputations(judge, red, blue, proportion_incumbents_red)
#     pR = proportion_incumbents_red # Rename for brevity
#     # A is a 2x2 matrix with elements aij
#     a11 = pR * (evaluate(judge, red, (1, 1)) - evaluate(judge, red, (1, 0))) - 1
#     a12 = (1 - pR) * (evaluate(judge, red, (0, 1)) - evaluate(judge, red, (0, 0)))
#     a21 = pR * (evaluate(judge, blue, (0, 1)) - evaluate(judge, blue, (0, 0)))
#     a22 = (1 - pR) * (evaluate(judge, blue, (1, 1)) - evaluate(judge, blue, (1, 0))) - 1
#     # b is a 2-vector with elements bi
#     b1 = pR * evaluate(judge, red, (1, 0)) + (1 - pR) * evaluate(judge, red, (0, 0))
#     b2 = pR * evaluate(judge, blue, (0, 0)) + (1 - pR) * evaluate(judge, blue, (1, 0))

#     det_A = a11 * a22 - a12 * a21
#     return ((a12 * b2 - a22 * b1), (a21 * b1 - a11 * b2)) ./ det_A
# end

function stationary_incumbent_reputations(judge, red, blue, proportion_incumbents_red)
    pR = proportion_incumbents_red # Rename for brevity
    # A is a 2x2 matrix with elements aij
    a11 = pR * (evaluate(judge, red, (1, 1)) - evaluate(judge, red, (1, 0))) - 1
    a12 = (1 - pR) * (evaluate(judge, red, (0, 1)) - evaluate(judge, red, (0, 0)))
    a21 = pR * (evaluate(judge, blue, (0, 1)) - evaluate(judge, blue, (0, 0)))
    a22 = (1 - pR) * (evaluate(judge, blue, (1, 1)) - evaluate(judge, blue, (1, 0))) - 1
    A = SA[a11 a12; a21 a22]
    # b is a 2-vector with elements bi
    b1 = pR * evaluate(judge, red, (1, 0)) + (1 - pR) * evaluate(judge, red, (0, 0))
    b2 = pR * evaluate(judge, blue, (0, 0)) + (1 - pR) * evaluate(judge, blue, (1, 0))
    b = SA[b1, b2]
    return -A\b
end

function stationary_mutant_reputations(judge, red, blue, R★, B★, proportion_incumbents_red)
    pR = proportion_incumbents_red
    pB = 1 - pR
    JRM11 = evaluate(judge, red, (1, 1))
    JRM10 = evaluate(judge, red, (1, 0))
    JRM01 = evaluate(judge, red, (0, 1))
    JRM00 = evaluate(judge, red, (0, 0))

    JBM11 = evaluate(judge, blue, (1, 1))
    JBM10 = evaluate(judge, blue, (1, 0))
    JBM01 = evaluate(judge, blue, (0, 1))
    JBM00 = evaluate(judge, blue, (0, 0))
    return (
        JRM00 * pB + JRM10 * pR - B★ * JRM00 * pB + B★ * JRM01 * pB - JRM10 * R★ * pR +
        JRM11 * R★ * pR,
        JBM10 * pB + JBM00 * pR - B★ * JBM10 * pB + B★ * JBM11 * pB - JBM00 * R★ * pR +
        JBM01 * R★ * pR,
    )
end

function stationary_reputations(judge, red, blue, red_mutant, blue_mutant, proportion_incumbents_red)
    R★, B★ = stationary_incumbent_reputations(judge, red, blue, proportion_incumbents_red)
    RM★, BM★ = stationary_mutant_reputations(judge, red_mutant, blue_mutant, R★, B★, proportion_incumbents_red)
    return R★, B★, RM★, BM★
end