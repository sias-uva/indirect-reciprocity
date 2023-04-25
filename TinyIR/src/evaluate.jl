# Evaluation semantics of social rules and execution/perception mistakes
function evaluate(rule::SVector{4}, (em, pm), (group_relation, reputation))
    function intention(rel, rep)
        rule[1] * (1 - rel) * (1 - rep) +
        rule[2] * rel * (1 - rep) +
        rule[3] * (1 - rel) * rep +
        rule[4] * rel * rep
    end
    perception(rel, rep) = (1 - em) * intention(rel, rep) + em * (1 - intention(rel, rep))
    function realisation(rel, rep)
        perception(rel, rep) * (1 - pm[1]) * (1 - pm[2]) +
        perception(1 - rel, rep) * (pm[1]) * (1 - pm[2]) +
        perception(rel, 1 - rep) * (1 - pm[1]) * (pm[2]) +
        perception(1 - rel, 1 - rep) * (pm[1]) * (pm[2])
    end
    return realisation(group_relation, reputation)
end

function evaluate(rule::SVector{8}, (em, pm), (group_relation, reputation, action))
    function intention(rel, rep, act)
        rule[1] * (1 - rel) * (1 - rep) * (1 - act) +
        rule[2] * rel * (1 - rep) * (1 - act) +
        rule[3] * (1 - rel) * rep * (1 - act) +
        rule[4] * rel * rep * (1 - act) +
        rule[5] * (1 - rel) * (1 - rep) * act +
        rule[6] * rel * (1 - rep) * act +
        rule[7] * (1 - rel) * rep * act +
        rule[8] * rel * rep * act
    end
    perception(rel, rep, act) = (1 - em) * intention(rel, rep, act) + em * (1 - intention(rel, rep, act))
    function realisation(rel, rep, act)
        perception(rel, rep, act) * (1 - pm[1]) * (1 - pm[2]) * (1 - pm[3]) +
        perception(1 - rel, rep, act) * pm[1] * (1 - pm[2]) * (1 - pm[3]) +
        perception(rel, 1 - rep, act) * (1 - pm[1]) * pm[2] * (1 - pm[3]) +
        perception(1 - rel, 1 - rep, act) * pm[1] * pm[2] * (1 - pm[3]) +
        perception(rel, rep, 1 - act) * (1 - pm[1]) * (1 - pm[2]) * pm[3] +
        perception(1 - rel, rep, 1 - act) * pm[1] * (1 - pm[2]) * pm[3] +
        perception(rel, 1 - rep, 1 - act) * (1 - pm[1]) * pm[2] * pm[3] +
        perception(1 - rel, 1 - rep, 1 - act) * pm[1] * pm[2] * pm[3]
    end
    return realisation(group_relation, reputation, action)
end

function evaluate(agent, info)
    agent_rule, agent_mistakes = agent
    evaluate(agent_rule, agent_mistakes, info)
end

function evaluate(judge, player, (group_relation, reputation))
    jrule, (jem, jpm) = judge
    prule, (pem, ppm) = player
    action = evaluate(prule, (pem, ppm), (group_relation, reputation))
    judgement = evaluate(jrule, (jem, jpm), (group_relation, reputation, action))
    return judgement
end