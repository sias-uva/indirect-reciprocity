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
player_perception_mistake_rate = SA[0.01, 0.01]
judge_perception_mistake_rate = SA[0.01, 0.01, 0.01]
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
perception_property_range = 0.001:0.001:0.5

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
    fig = Figure(; resolution=(600, 575))
    cmap = cgrad(:Hiroshige, 4; rev=true, categorical=true)
    begin
        # TOPLEFT BOX: Player execution oopsie rate
        pem_ps = (merge(p, (; maj_em=rate, min_em=rate)) for rate in property_range)
        df_vector = map(pem_ps) do p
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
            df.pem .= p.min_em
            df
        end
        @show df_vector[1].cooperativeness[1]
        res = vcat(df_vector...)
        begin
            ax_pem = Axis(
                fig[1, 1];
                title="Player execution",
                # xlabel="Mistake rate",
                ylabel="Cooperativeness",
                # limits=((0, 0.5), (0, 1))
            )
            # for (i, norm_int) in enumerate(norm_ints)
            #     data = subset(res, :norm => ByRow(==(norm_int)))
            #     ess_cutoff = argmin(data.is_ess)
            #     vlines!(
            #         ax_pem,
            #         data[ess_cutoff, :pem],
            #         ymax = data[ess_cutoff, :cooperativeness] - 0.04;
            #         linewidth=1,
            #         label="$norm_int",
            #         linestyle=:dash,
            #         color=:black
            #     )
            # end
            for (i, norm_int) in enumerate(norm_ints)
                data = subset(res, :norm => ByRow(==(norm_int)))
                ess_cutoff = argmin(data.is_ess)
                lines!(
                    ax_pem,
                    data[1:ess_cutoff, :pem],
                    data[1:ess_cutoff, :cooperativeness];
                    linewidth,
                    label="$norm_int",
                    color=cmap[i],
                )
                lines!(
                    ax_pem,
                    data[(ess_cutoff + 1):end, :pem],
                    data[(ess_cutoff + 1):end, :cooperativeness];
                    linewidth,
                    label="$norm_int",
                    color=cmap[i],
                    linestyle=:dash,
                )
                scatter!(
                    ax_pem,
                    data[ess_cutoff, :pem],
                    data[ess_cutoff, :cooperativeness];
                    color=cmap[i],
                    marker=hollow_circle,
                    markersize=5,
                    label="$norm_int",
                )
            end
            # axislegend(ax_pem)
            # legend => [norms => colours, (solid, dash) => (ess, not_ess)]
        end
    end

    # TOPRIGHT: Judge execution rate
    begin
        jem_ps = (merge(p, (; judge_em=rate)) for rate in property_range)
        df_vector = map(jem_ps) do p
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
            df.judge_em .= p.judge_em
            df
        end
        @show df_vector[1].cooperativeness[1]
        res = vcat(df_vector...)
        begin
            ax_jem = Axis(
                fig[1, 2];
                title="Judge execution",
                # # xlabel="Mistake rate",
                # ylabel="Cooperativeness"
            )
            # for (i, norm_int) in enumerate(norm_ints)
            #     data = subset(res, :norm => ByRow(==(norm_int)))
            #     ess_cutoff = argmin(data.is_ess)
            #     vlines!(
            #         ax_jem,
            #         data[ess_cutoff, :judge_em],
            #         ymax = data[ess_cutoff, :cooperativeness] - 0.04;
            #         linewidth=1,
            #         label="$norm_int",
            #         linestyle=:dash,
            #         color=:black
            #     )
            # end
            for (i, norm_int) in enumerate(norm_ints)
                data = subset(res, :norm => ByRow(==(norm_int)))
                ess_cutoff = argmin(data.is_ess)
                lines!(
                    ax_jem,
                    data[1:ess_cutoff, :judge_em],
                    data[1:ess_cutoff, :cooperativeness];
                    linewidth,
                    label="$norm_int",
                    color=cmap[i],
                )
                lines!(
                    ax_jem,
                    data[(ess_cutoff + 1):end, :judge_em],
                    data[(ess_cutoff + 1):end, :cooperativeness];
                    linewidth,
                    label="$norm_int",
                    color=cmap[i],
                    linestyle=:dash,
                )
                scatter!(
                    ax_jem,
                    data[ess_cutoff, :judge_em],
                    data[ess_cutoff, :cooperativeness];
                    color=cmap[i],
                    marker=hollow_circle,
                    label="$norm_int",
                    markersize=5,
                )
            end
        end
    end

    # BOTTOMLEFT: Judge perception mistake rate
    begin
        jpm_ps = (merge(p, (; judge_pm=SA[0, rate, 0])) for rate in property_range)
        df_vector = map(jpm_ps) do p
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
            df.judge_pm .= p.judge_pm[2]
            df
        end
        @show df_vector[1].cooperativeness[1]
        res = vcat(df_vector...)
        begin
            ax_jpm = Axis(
                fig[2, 1];
                title="Judge reputation perception",
                xlabel="Mistake rate",
                ylabel="Cooperativeness",
            )
            # for (i, norm_int) in enumerate(norm_ints)
            #     data = subset(res, :norm => ByRow(==(norm_int)))
            #     ess_cutoff = argmin(data.is_ess)
            #     vlines!(
            #         ax_jpm,
            #         data[ess_cutoff, :judge_pm],
            #         ymax = data[ess_cutoff, :cooperativeness] - 0.04;
            #         linewidth=1,
            #         label="$norm_int",
            #         linestyle=:dash,
            #         color=:black
            #     )
            # end
            for (i, norm_int) in enumerate(norm_ints)
                data = subset(res, :norm => ByRow(==(norm_int)))
                ess_cutoff = argmin(data.is_ess)
                # @show norm_int, ess_cutoff
                lines!(
                    ax_jpm,
                    data[1:ess_cutoff, :judge_pm],
                    data[1:ess_cutoff, :cooperativeness];
                    linewidth,
                    label="$norm_int",
                    color=cmap[i],
                )
                lines!(
                    ax_jpm,
                    data[(ess_cutoff + 1):end, :judge_pm],
                    data[(ess_cutoff + 1):end, :cooperativeness];
                    label="$norm_int",
                    linewidth,
                    color=cmap[i],
                    linestyle=:dash,
                )
                scatter!(
                    ax_jpm,
                    data[ess_cutoff, :judge_pm],
                    data[ess_cutoff, :cooperativeness];
                    color=cmap[i],
                    marker=hollow_circle,
                    label="$norm_int",
                    markersize=5,
                )
            end
        end
    end
    # BOTTOMLEFT: Judge perception mistake rate
    begin
        action_perception_ps = (
            merge(p, (; judge_pm=SA[0, 0, rate])) for rate in property_range
        )
        df_vector = map(action_perception_ps) do p
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
            df.judge_pm .= p.judge_pm[3]
            df
        end
        @show df_vector[1].cooperativeness[1]
        res = vcat(df_vector...)
        begin
            ax_action_perception = Axis(
                fig[2, 2];
                title="Judge action perception",
                xlabel="Mistake rate",
                # ylabel="Cooperativeness"
            )
            # for (i, norm_int) in enumerate(norm_ints)
            #     data = subset(res, :norm => ByRow(==(norm_int)))
            #     ess_cutoff = argmin(data.is_ess)
            #     vlines!(
            #         ax_action_perception,
            #         data[ess_cutoff, :judge_pm],
            #         ymax = data[ess_cutoff, :cooperativeness] - 0.04;
            #         linewidth=1,
            #         label="$norm_int",
            #         linestyle=:dash,
            #         color=:black
            #     )
            # end
            for (i, norm_int) in enumerate(norm_ints)
                data = subset(res, :norm => ByRow(==(norm_int)))
                ess_cutoff = argmin(data.is_ess)
                # @show norm_int, ess_cutoff
                lines!(
                    ax_action_perception,
                    data[1:ess_cutoff, :judge_pm],
                    data[1:ess_cutoff, :cooperativeness];
                    linewidth,
                    label="$norm_int",
                    color=cmap[i],
                )
                lines!(
                    ax_action_perception,
                    data[(ess_cutoff + 1):end, :judge_pm],
                    data[(ess_cutoff + 1):end, :cooperativeness];
                    linewidth,
                    label="$norm_int",
                    color=cmap[i],
                    linestyle=:dash,
                )
                scatter!(
                    ax_action_perception,
                    data[ess_cutoff, :judge_pm],
                    data[ess_cutoff, :cooperativeness];
                    color=cmap[i],
                    marker=hollow_circle,
                    label="$norm_int",
                    markersize=5,
                )
            end
        end
    end
    linkaxes!(ax_jem, ax_jpm, ax_action_perception, ax_pem)

    # Make custom legend:
    markersize = 7.5
    marker = Polygon(decompose(Point2f, Circle(Point2f(0), 1)))
    style_elements = [
        LineElement(; color=:black, linestyle, linewidth) for linestyle in (:solid, :dash)
    ]
    color_elements = let
        vect = [
            MarkerElement(; color, marker, markersize, strokewidth=1) for
            color in getindex.(Ref(cmap), 1:3)
        ]
        vect
    end
    style_labels = ["Stable", "Unstable"]
    color_labels = ["SH", "SJ", "SS"]
    legend = Legend(
        fig[3, :],
        [style_elements, color_elements],
        [style_labels, color_labels],
        ["Stability", "Norm"];
        # nbanks=2,
        # titleposition=:left,
        orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
    )
    resize_to_layout!(fig)
    for filetype in ("png", "pdf")
        save("figures/interdisciplinary/alternative_setups_grid.$filetype", fig)
    end
    fig
end
