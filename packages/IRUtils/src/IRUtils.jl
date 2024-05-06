module IRUtils

using IR
using StaticArrays
using DataFrames

export get_agents,
    find_ESS,
    split_norm,
    generate_quadrant_df,
    is_fair,
    categorise_strategy_old,
    categorise_strategy,
    whichmax

# analytical
function get_agents(norm_i, rs_i, bs_i; p)
    norm = iNorm(norm_i)
    judge = Agent(norm, p.judge_em, p.judge_pm)
    majority_strat = iStrategy(rs_i)
    majority = Agent(majority_strat, p.maj_em, p.maj_pm)
    minority_strat = iStrategy(bs_i)
    minority = Agent(minority_strat, p.min_em, p.min_pm)
    return judge, majority, minority
end

function find_ESS(p, kwargs...)
    norm_ints = 0:255
    strategy_ints = 0:15
    df = rename!(
        DataFrame(Iterators.product(norm_ints, strategy_ints, strategy_ints)),
        [:norm, :majority_strat, :minority_strat],
    )
    df[!, :is_ess] = map(eachrow(df)) do row
        judge, majority, minority = get_agents(row...; p)
        is_ESS(judge, majority, minority, p.prop_maj, p.utilities; kwargs...)
    end
    return df
end

function split_norm(i)
    norm = iNorm(i)
    outgroup_norm = evalpoly(2, reshape(norm[1, :, :], 4))
    ingroup_norm = evalpoly(2, reshape(norm[2, :, :], 4))
    return (; ingroup_norm, outgroup_norm)
end

function generate_quadrant_df(_df; p)
    df = deepcopy(_df)
    info_df = DataFrame(
        [
            :majority_rep,
            :minority_rep,
            :majority_payoff,
            :minority_payoff,
            :fairness,
            :cooperation,
            :prr,
            :pbr,
            :prd,
            :pbd,
        ] .=> Ref(Float64[]),
    )
    quadrants = Int8[]
    foreach(eachrow(df)) do row
        n, r, b, _ = row
        judge, majority, minority = get_agents(n, r, b; p)
        majority_rep, minority_rep = stationary_incumbent_reputations(
            judge, majority, minority, p.prop_maj
        )
        majority_payoff, minority_payoff = incumbent_payoffs(
            majority, minority, majority_rep, minority_rep, p.prop_maj, p.utilities
        )
        prr = p_receives(majority, minority, majority_rep, p.prop_maj)
        prd = p_donates(majority, majority_rep, minority_rep, p.prop_maj)
        pbr = p_receives(minority, majority, minority_rep, 1 - p.prop_maj)
        pbd = p_donates(minority, minority_rep, majority_rep, 1 - p.prop_maj)
        fairness = let
            lower, higher = minmax(majority_payoff, minority_payoff)
            # if (n == 242 && r == 13 & b == 0)
            #     @show lower higher
            #     println(lower/higher)
            # end
            lower / higher
        end
        cooperation = p.prop_maj * prd + (1 - p.prop_maj) * pbd
        push!(
            info_df,
            (
                majority_rep,
                minority_rep,
                majority_payoff,
                minority_payoff,
                fairness,
                cooperation,
                prr,
                pbr,
                prd,
                pbd,
            ),
        )
        push!(quadrants, (cooperation < 0.5) + 2(fairness < 0.5))
    end
    df = hcat(df, info_df, quadrants)
    rename!(df, :x1 => :quadrant)
    subset!(df, :is_ess)
    # unique!(df, [:majority_payoff, :minority_payoff, :majority_rep, :minority_rep])
    return df
end

function is_fair(x)
    sr = reshape(iNorm(x), Size(8))
    return all(1:2:7) do i
        sr[i] == sr[i + 1]
    end
end

function categorise_strategy(i)
    i == 0 && return 0 # AllD
    !is_fair(i) && return 1 # Discriminatory
    return 2 # Group-agnostic
end

function categorise_strategy_old(i)
    Base.depwarn(
        "`categorise_strategy_old(i)` is deprecated, use `categorise_strategy(i)` instead and deal with the difference in output.",
        :categorise_strategy,
    )
    i == 0 && return 1 # AllD
    is_fair(i) && return 0 # Group-agnostic
    return 2 # Discriminatory
end

whichmax(x, y) = argmax((x, y)) - 1

end # module IRUtils
