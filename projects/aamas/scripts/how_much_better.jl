using IR
using IRUtils
using StaticArrays
using DataFrames
using ColorSchemes
using Tidier
using StatsBase

function get_next_best_delta(
    judge, majority::Player, minority::Player, prop_majority, utilities
)
    majority_minimum = Inf
    minority_minimum = Inf
    majority_whichmin = []
    minority_whichmin = []
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
            curr_majority_minimum = payoff_majority - payoff_majority_mutant
            if curr_majority_minimum == majority_minimum
                push!(majority_whichmin, i)
            elseif curr_majority_minimum < majority_minimum
                majority_minimum = curr_majority_minimum
                majority_whichmin = [i]
            end
        end
        if mutant_strategy != minority.rule
            curr_minority_minimum = payoff_minority - payoff_minority_mutant
            if curr_minority_minimum == minority_minimum
                push!(minority_whichmin, i)
            elseif curr_minority_minimum < minority_minimum
                minority_minimum = curr_minority_minimum
                minority_whichmin = [i]
            end
        end
    end
    return (majority_minimum, minority_minimum, majority_whichmin, minority_whichmin)
end

player_execution_mistake_rate = 0.01
judge_execution_mistake_rate = 0.01
player_perception_mistake_rate = SA[0.00, 0.00]
judge_perception_mistake_rate = SA[0.00, 0.00, 0.00]
proportion_incumbents_majority = 0.9
utilities = SA[5, 5, 1, 1]

p = (;
    maj_em=player_execution_mistake_rate,
    min_em=player_execution_mistake_rate,
    judge_em=judge_execution_mistake_rate,
    maj_pm=player_perception_mistake_rate,
    min_pm=player_perception_mistake_rate,
    judge_pm=judge_perception_mistake_rate,
    prop_maj=proportion_incumbents_majority,
    utilities=utilities,
);

filters = [
# [:majority_strat, :minority_strat] => ByRow((r, b) -> is_fair(r) && (b == 0 || !is_fair(b)))
];

df_delta = let
    df = find_ESS(p)
    subset!(df, :is_ess)

    df_all_invaders =
        nbd =
            get_next_best_delta.(
                Agent.(iNorm.(df.norm), Ref(p.judge_em), Ref(p.judge_pm)),
                Agent.(iStrategy.(df.majority_strat), Ref(p.maj_em), Ref(p.maj_pm)),
                Agent.(iStrategy.(df.minority_strat), Ref(p.min_em), Ref(p.min_pm)),
                Ref(p.prop_maj),
                Ref(utilities),
            )
    df[:, :majority_minimum] = first.(first.(nbd))
    df[:, :minority_minimum] = last.(first.(nbd))
    df[:, :majority_whichmin] = first.(last.(nbd))
    df[:, :minority_whichmin] = last.(last.(nbd))
    select!(df, Not(:is_ess))
    df
end

subset(
    sort(df_delta, :majority_minimum),
    :norm => ByRow(is_fair),
    :majority_strat => ByRow(!=(0)),
)
