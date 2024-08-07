# Unfavourable setups, shown as a grid:
# - Inability to consistently cooperate when intending to, how far can we go before everything falls apart?
# - Inability of norm to consistently correctly assess
# - 

# x-axis: rate of thing that is changing up to 0.5
# y-axis: cooperativeness (talk about fairness in text and put plot in appendix)
# In each setup in the grid, the 4 named norms (SS, SJ, SH, IS) each have a unique colour.
# The place where it stops being an ESS is marked with a scatter point

# Test log scale for rates, magnitudes are what matter (I imagine)

using IR
using StaticArrays
using DataFrames
using CairoMakie
using GeometryBasics

include("../norms.jl") # norms, simple_norms|
include("../misc.jl")

my_norm_names = Dict(192 => "SH", 195 => "SJ", 243 => "SS")

# Baseline properties
player_execution_mistake_rate = 0.01
judge_execution_mistake_rate = 0.01
player_perception_mistake_rate = SA[0.00, 0.00]
judge_perception_mistake_rate = SA[0.00, 0.00, 0.00]
proportion_incumbents_majority = 0.9
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

# Property ranges
property_range = 0.01:0.0001:0.5

# Setups to check
norm_ints = (192, 195, 243)
# norm_ints = (195, 243)
# norm_ints = (243,)
strategy_ints = (12,)

# Marker
p_big = decompose(Point2f, Circle(Point2f(0), 1))
p_small = decompose(Point2f, Circle(Point2f(0), 0.5))
hollow_circle = Polygon(p_big, [p_small])

# Global plot settings
linewidth = 2.0

let
    fig = Figure(; resolution=(600, 500))
    cmap = cgrad(:Hiroshige, 4; rev=true, categorical=true)
    # minority_em_ps = (merge(p, (; min_em=rate, judge_pm=SA[0.0, 0.1, 0])) for rate in property_range)
    minority_em_ps = (merge(p, (; min_em=rate)) for rate in property_range)
    df_vector = map(minority_em_ps) do p
        df = rename!(
            DataFrame(Iterators.product(norm_ints, strategy_ints, strategy_ints)),
            [:norm, :majority_strat, :minority_strat],
        )
        df.is_ess = map(eachrow(df)) do row
            judge, majority, minority = get_agents(row...; p)
            is_ESS(judge, majority, minority, p.prop_maj, p.utilities)
        end
        coop_fair = map(eachrow(df)) do row
            n, r, b, _ = row
            judge, majority, minority = get_agents(n, r, b; p)
            majority_rep, minority_rep = stationary_incumbent_reputations(
                judge, majority, minority, p.prop_maj
            )
            majority_payoff, minority_payoff = incumbent_payoffs(
                majority,
                minority,
                majority_rep,
                minority_rep,
                p.prop_maj,
                p.utilities,
            )
            prr = p_receives(majority, minority, majority_rep, p.prop_maj)
            prd = p_donates(majority, majority_rep, minority_rep, p.prop_maj)
            pbr = p_receives(minority, majority, minority_rep, 1 - p.prop_maj)
            pbd = p_donates(minority, minority_rep, majority_rep, 1 - p.prop_maj)
            fairness = let
                lower, higher = minmax(majority_payoff, minority_payoff)
                lower / higher
            end
            cooperation = p.prop_maj * prd + (1 - p.prop_maj) * pbd
            cooperation, fairness
        end
        df.cooperativeness = first.(coop_fair)
        df.fairness = last.(coop_fair)
        df.minority_em .= p.min_em
        df
    end
    res = vcat(df_vector...)
    ticks = 0:0.25:1
    ax_bem = Axis(
        fig[1, 1];
        xlabel="Cooperativeness",
        ylabel="Fairness",
        xticks=ticks,
        yticks=ticks,
        title="",
        titlealign=:left,
        aspect=DataAspect(),
    )
    for (i, norm_int) in enumerate(norm_ints)
        data = subset(res, :norm => ByRow(==(norm_int)))
        ess_cutoff = if all(data.is_ess)
            nrow(data)
        else
            argmin(data.is_ess)
        end
        @show ess_cutoff
        println(data[ess_cutoff, :minority_em])
        println(data[ess_cutoff, :fairness])
        lines!(
            ax_bem,
            data[1:ess_cutoff, :cooperativeness],
            data[1:ess_cutoff, :fairness];
            linewidth,
            label="$norm_int",
            color=cmap[i],
        )
        lines!(
            ax_bem,
            data[(ess_cutoff + 1):end, :cooperativeness],
            data[(ess_cutoff + 1):end, :fairness];
            linewidth,
            label="$norm_int",
            color=cmap[i],
            linestyle=:dash,
        )
        scatter!(
            ax_bem,
            data[ess_cutoff, :cooperativeness],
            data[ess_cutoff, :fairness];
            color=cmap[i],
            marker=hollow_circle,
            label="$norm_int",
            markersize=5,
        )
    end
    offset = 0.025
    limits!(ax_bem, (0 - offset, 1 + offset), (0 - offset, 1 + offset))
    resize_to_layout!(fig)
    for filetype in ("png", "pdf")
        save(
            "figures/interdisciplinary/unfair_alternative_setups_coop_fairness.$filetype",
            fig,
        )
    end
    fig
end
