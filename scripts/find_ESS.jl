using IR
using StaticArrays
using DataFrames
using CairoMakie

parameters = (;
    player_execution_mistake_rate = 0.01,
    judge_execution_mistake_rate = 0.0,
    player_perception_mistake_rate = 0.0,
    judge_perception_mistake_rate = 0.0,
    proportion_incumbents_red = 0.9,
    utilities = SA[2.0, 2, 1, 1],
)

function find_ESS(jem, jpm, rem, rpm, bem, bpm, pR, utilities)
    norm_ints = 0:255
    strategy_ints = 0:15
    df = rename!(
        DataFrame(Iterators.product(norm_ints, strategy_ints, strategy_ints)),
        [:norm, :red_strat, :blue_strat],
    )
    df[!, :is_ess] = map(eachrow(df)) do row
        norm_i, rs_i, bs_i = row
        norm = iNorm(norm_i)
        judge = Agent(norm, jem, jpm)
        red_strat = iStrategy(rs_i)
        red = Agent(red_strat, rem, rpm)
        blue_strat = iStrategy(bs_i)
        blue = Agent(blue_strat, bem, bpm)
        is_ESS(judge, red, blue, pR, utilities)
    end
    return df
end

function main(parameters)
    pem, jem, ppm, jpm, pR, utils = parameters
    df = find_ESS(jem, jpm, pem, ppm, pem, ppm, pR, utils)
    subset!(df, :is_ess)


end

main(parameters)
