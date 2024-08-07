using IR
using StaticArrays
using DataFrames
using CairoMakie
using GeometryBasics
using ColorSchemes
using Format
using CSV
using Tidier

# CairoMakie.activate!()
using IRUtils

begin
    player_execution_mistake_rate = 0.01
    judge_execution_mistake_rate = 0.01
    player_perception_mistake_rate = SA[0.00, 0.00]
    judge_perception_mistake_rate = 0.00
    proportion_incumbents_majority = 0.9
    utilities = SA[10, 10, 1, 1]

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

    filters = [
    # [:majority_strat, :minority_strat] => ByRow((r, b) -> is_fair(r) && (b == 0 || !is_fair(b)))
]
end

df = generate_quadrant_df(find_ESS(p); p)
df_rl = CSV.read(
    "projects/aamas/data/rl_data_10_0:255.csv", DataFrame; stripwhitespace=true
)

let
    begin # Data for plot
        df_egt = copy(df)
        transform!(
            df_egt,
            [:majority_strat, :minority_strat] .=>
                ByRow(categorise_strategy_old) .=>
                    [:majority_strat_category, :minority_strat_category],
        )
        subset!(df_egt, :majority_strat => ByRow(!=(0)))
        subset!(
            df_egt,
            :norm => ByRow(in([150, 192, 195, 243])),
            :majority_strat => ByRow(in([12, 9])),
        )
        select!(df_egt, :norm, :cooperation, :fairness)
    end

    ticks = 0:0.25:1
    fig = Figure(; resolution=(600, 600))
    ax = Axis(
        fig[1, 1];
        xlabel="Cooperativeness",
        ylabel="Fairness",
        # xticks=ticks,
        # yticks=ticks,
        title="",
        titlealign=:left,
        # aspect=DataAspect(),
    )
    offset = 0.025
    xlims = (0 - offset, 1 + offset)
    ylims = (0.9 - offset, 1 + offset)
    # limits!(ax, xlims, ylims)

    scatter_settings = (markersize=20, strokewidth=1.2)
    markers = [:circle, :rect, :cross]
    cmap = cgrad(:Egypt, 4; categorical=true)

    df_new = @chain df begin
        groupby(:norm)
        combine(sdf -> first(sort(sdf, :cooperation; rev=true)))
        subset([:majority_strat, :minority_strat] => ByRow((x, y) -> !(x == y == 0)))
        select(:norm, :cooperation, :fairness)
        leftjoin(df_rl; on=:norm, makeunique=true)
        subset(:norm => ByRow(in((150, 192, 195, 243))))
        rename(:cooperation_1 => :cooperation_rl, :fairness_1 => :fairness_rl)
    end
    df_egt = select(df_new, :norm, :cooperation, :fairness)
    df_rl2 = select(
        df_new, :norm, :cooperation_rl => :cooperation, :fairness_rl => :fairness
    )

    df_new.coop_dir = (df_new.cooperation_rl .- df_new.cooperation) ./ 2
    df_new.fairness_dir = (df_new.fairness_rl .- df_new.fairness) ./ 2
    arrows!(
        ax,
        df_new.cooperation,
        df_new.fairness,
        df_new.coop_dir,
        df_new.fairness_dir;
        linewidth=1.2,
    )
    arrows!(
        ax,
        df_new.cooperation_rl,
        df_new.fairness_rl,
        .-df_new.coop_dir,
        .-df_new.fairness_dir;
        arrowsize=0,
        linewidth=1.2,
    )
    for (i, df) in enumerate((df_egt, df_rl2))
        df = subset(df, :norm => ByRow(in((150, 192, 195, 243))))
        scatter!(
            ax,
            df.cooperation,
            df.fairness;
            marker=markers[i],
            color=sortperm(df.norm),
            scatter_settings...,
            colormap=cmap,
            alpha=1,
        )
    end

    marker = Polygon(decompose(Point2f, Circle(Point2f(0), 1))) # <- AT THIS POINT, THE ANSWER WAS NO
    marker_elements = [
        MarkerElement(; color=:grey, marker, scatter_settings...) for marker in markers
    ]
    color_elements = [
        MarkerElement(; color, marker, markersize=10, strokewidth=1) for
        color in getindex.(Ref(cmap), 1:4)
    ]
    marker_labels = ["EGT", "RL"]
    color_labels = ["150", "Shunning", "Stern Judging", "Simple Standing"]
    legend = Legend(
        fig[2, 1],
        [marker_elements, color_elements],#
        [marker_labels, color_labels],#
        ["Model", "Norm"];
        nbanks=2,
        orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
    )
    resize_to_layout!(fig)
    for filetype in ("pdf", "png")
        save("projects/aamas/figures/egt-rl-comparison.$filetype", fig)
    end
    fig
end

# Numerically tables:
begin
    df_new = @chain df begin
        groupby(:norm)
        combine(sdf -> first(sort(sdf, :cooperation; rev=true)))
        subset([:majority_strat, :minority_strat] => ByRow((x, y) -> !(x == y == 0)))
        select(:norm, :cooperation, :fairness)
        leftjoin(df_rl; on=:norm, makeunique=true)
        rename(:cooperation_1 => :cooperation_rl, :fairness_1 => :fairness_rl)
        transform(
            [:cooperation, :fairness] .=> ByRow(>(0.5)) .=> [:x_quadrant, :y_quadrant]
        )
        groupby([:x_quadrant, :y_quadrant])
        transform(
            [:cooperation, :cooperation_rl] => ByRow((x, y) -> y - x) => :coop_diff;
            ungroup=false,
        )
        transform(
            [:fairness, :fairness_rl] => ByRow((x, y) -> y - x) => :fairness_diff;
            ungroup=false,
        )
        combine([:coop_diff, :fairness_diff] .=> mean .=> [:coop_diff, :fairness_diff])
        transform(
            [:x_quadrant, :y_quadrant] => ByRow((a, b) -> evalpoly(2, (a, b))) => :quadrant
        )
    end
    sort!(df_new, :quadrant)
    df_new.quadrant_name .= ["bottom_left", "bottom_right", "top_left", "top_right"]
    select!(df_new, :quadrant_name, :coop_diff, :fairness_diff)
    push!(df_new, ["total", mean(df_new.coop_diff), mean(df_new.fairness_diff)])
    df_new
end

begin
    df_new_no_zeros = @chain df begin
        subset(:fairness => ByRow(!=(0)))
        groupby(:norm)
        combine(sdf -> first(sort(sdf, :cooperation; rev=true)))
        subset([:majority_strat, :minority_strat] => ByRow((x, y) -> !(x == y == 0)))
        select(:norm, :cooperation, :fairness)
        leftjoin(df_rl; on=:norm, makeunique=true)
        rename(:cooperation_1 => :cooperation_rl, :fairness_1 => :fairness_rl)
        transform(
            [:cooperation, :fairness] .=> ByRow(>(0.5)) .=> [:x_quadrant, :y_quadrant]
        )
        groupby([:x_quadrant, :y_quadrant])
        transform(
            [:cooperation, :cooperation_rl] => ByRow((x, y) -> y - x) => :coop_diff;
            ungroup=false,
        )
        transform(
            [:fairness, :fairness_rl] => ByRow((x, y) -> y - x) => :fairness_diff;
            ungroup=false,
        )
        combine([:coop_diff, :fairness_diff] .=> mean .=> [:coop_diff, :fairness_diff])
        transform(
            [:x_quadrant, :y_quadrant] => ByRow((a, b) -> evalpoly(2, (a, b))) => :quadrant
        )
    end
    sort!(df_new_no_zeros, :quadrant)
    df_new_no_zeros.quadrant_name .= ["bottom_right", "top_left", "top_right"]
    select!(df_new_no_zeros, :quadrant_name, :coop_diff, :fairness_diff)
    push!(
        df_new_no_zeros,
        ["total", mean(df_new_no_zeros.coop_diff), mean(df_new_no_zeros.fairness_diff)],
    )
    df_new_no_zeros
end

df_diff = @chain df begin
    groupby(:norm)
    combine(sdf -> first(sort(sdf, :cooperation; rev=true)))
    subset([:majority_strat, :minority_strat] => ByRow((x, y) -> !(x == y == 0)))
    select(:norm, :cooperation, :fairness)
    leftjoin(df_rl; on=:norm, makeunique=true)
    rename(:cooperation_1 => :cooperation_rl, :fairness_1 => :fairness_rl)
    transform([:cooperation, :cooperation_rl] => ByRow((x, y) -> y - x) => :coop_diff)
    transform([:fairness, :fairness_rl] => ByRow((x, y) -> y - x) => :fairness_diff)
end

@chain df_diff begin
    subset(:norm => ByRow(==(243)))
end
