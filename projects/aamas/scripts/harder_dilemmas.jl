using IR
using StaticArrays
using DataFrames
using ColorSchemes
using Tidier
using StatsBase

# CairoMakie.activate!()

include("../../wip/scripts/norms.jl") # norms, simple_norms|
include("../../wip/scripts/misc.jl")

function all_invaders(judge, majority::Player, minority::Player, prop_majority, utilities)
    majority_invaders = Int64[]
    minority_invaders = Int64[]
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
            (payoff_majority >= payoff_majority_mutant) ||
                push!(majority_invaders, @evalpoly(2, majority_mutant.rule...))
        end
        if mutant_strategy != minority.rule
            payoff_minority >= payoff_minority_mutant ||
                push!(minority_invaders, @evalpoly(2, minority_mutant.rule...))
        end
    end
    return (majority_invaders, minority_invaders)
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

df = find_ESS(p)
subset(df, :is_ess)

# Norm colours by what do they discriminate vs (none, one , other, both),

begin
    for (i, benefit) in enumerate((1.05, 5, 15))
        df_all_invaders =
            all_invaders.(
                Agent.(iNorm.(df.norm), Ref(p.judge_em), Ref(p.judge_pm)),
                Agent.(iStrategy.(df.majority_strat), Ref(p.maj_em), Ref(p.maj_pm)),
                Agent.(iStrategy.(df.minority_strat), Ref(p.min_em), Ref(p.min_pm)),
                Ref(p.prop_maj),
                Ref([benefit, benefit, 1, 1]),
            )
        benefit_name = ("hard", "normal", "easy")[i]
        df[:, "majority_invaders_$benefit_name"] = first.(df_all_invaders)
        df[:, "minority_invaders_$benefit_name"] = last.(df_all_invaders)
    end
    # filter not ess and then is ess between "hard" and "normal", then "normal"
    # and "easy", in each case, identify the invaders that are "weakened" enough
    # to permit the stability of the incumbents.

    # @chain df begin
    #     transform(["majority_invaders_1.05", ])
    # end
    df
end

df_sub1 = @chain df begin
    subset(
        [:majority_invaders_hard, :minority_invaders_hard] =>
            ByRow((x, y) -> !(x == y == Int64[])),
        [:majority_invaders_normal, :minority_invaders_normal] =>
            ByRow((x, y) -> (x == y == Int64[])),
    )
    select(
        :norm,
        :majority_strat,
        :minority_strat,
        :majority_invaders_hard,
        :minority_invaders_hard,
    )
    transform(:norm => ByRow(split_norm) => AsTable)
end

df_sub2 = @chain df begin
    subset(
        [:majority_invaders_normal, :minority_invaders_normal] =>
            ByRow((x, y) -> !(x == y == Int64[])),
        [:majority_invaders_easy, :minority_invaders_easy] =>
            ByRow((x, y) -> (x == y == Int64[])),
    )
    select(
        :norm,
        :majority_strat,
        :minority_strat,
        :majority_invaders_normal,
        :minority_invaders_normal,
    )
    transform(:norm => ByRow(split_norm) => AsTable)
end

subset(df_sub1, :minority_strat => ByRow(!=(0)))
subset(df_sub2, :minority_strat => ByRow(!=(0)))

countmap(vcat(filter(!isempty, df_sub2.minority_invaders_normal)...))

foreach(println, groupby(df_sub1, :minority_invaders_hard))
foreach(println, groupby(df_sub1, :majority_invaders_hard))
foreach(println, groupby(df_sub2, :minority_invaders_normal))
foreach(println, groupby(df_sub2, :majority_invaders_normal))

# As it turns out, there is no overarching pattern in what invades every norm,
# however, depending on whether the majority or minority were invaded, each
# ingroup or outgroup norm appears to have a particular strategy that invades no
# matter the other group's strategy.
