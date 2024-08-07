using IR
using StaticArrays
using DataFrames
using CairoMakie
using ColorSchemes
using Format

CairoMakie.activate!()

include("../norms.jl") # norms, simple_norms|
include("../misc.jl")

player_execution_mistake_rate = 0.01
judge_execution_mistake_rate = 0.01
player_perception_mistake_rate = SA[0.00, 0.00]
judge_perception_mistake_rate = 0.00
proportion_incumbents_majority = 0.8
utilities = SA[2, 2, 1, 1]

p = (;
    maj_em=player_execution_mistake_rate,
    min_em=player_execution_mistake_rate,
    judge_em=judge_execution_mistake_rate,
    maj_pm=player_perception_mistake_rate,
    min_pm=player_perception_mistake_rate,
    judge_pm=judge_perception_mistake_rate,
    prop_maj=proportion_incumbents_majority,
    utilities=utilities,
)

filters = [:norm => ByRow(is_fair)]

df = find_ESS(p)
quadrant_df = generate_quadrant_df(df; p)

subset(quadrant_df, :is_ess, filters...)

subset(
    df,
    :is_ess,
    [:majority_strat, :minority_strat] => (x, y) -> .!(x .== y .== 0),
    filters...,
)
