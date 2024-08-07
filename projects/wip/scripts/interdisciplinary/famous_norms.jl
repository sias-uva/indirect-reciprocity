using IR
using IRUtils
using StaticArrays
using DataFrames
using CairoMakie
using GeometryBasics
using ColorSchemes
using Format
using Tidier

# CairoMakie.activate!()

# include("../norms.jl") # norms, simple_norms|
# include("../misc.jl")

begin
    player_execution_mistake_rate = 0.01
    judge_execution_mistake_rate = 0.01
    player_perception_mistake_rate = SA[0.00, 0.00]
    judge_perception_mistake_rate = SA[0.00, 0.00, 0.00]
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

    # Norm colours by what do they discriminate vs (none, one , other, both),

    df = find_ESS(p)
end

famous_norms = (8, 9, 12, 13)
df_famous_norms_minority = @chain df begin
    generate_quadrant_df(; p=p)
    # select(:norm, :majority_strat, :minority_strat, :cooperation, :fairness)
    transform(:norm => ByRow(split_norm) => AsTable)
    subset([:ingroup_norm, :outgroup_norm] .=> ByRow(in(famous_norms)))
    groupby(:norm)
    combine(sdf -> first(sort(sdf, :cooperation; rev=true)))
    # sort(:cooperation, rev=true)
    # subset(:outgroup_norm => ByRow(==(12)))
    select(:ingroup_norm, :outgroup_norm, :pbr)
    unstack(:outgroup_norm, :pbr)
    # select(:ingroup_norm, reverse(["12", "8", "9", "13"]))
end

transpose(Matrix{Float64}(df_famous_norms_minority[:, 2:end]))

let
    # tick_labels = ["Shunning", "SternJudging", "ImageScore", "SimpleStanding"]
    tick_labels = ["SJ", "SS", "SH", "IS"]
    # ["SH", "SJ", "IS", "SS"] # Original order 

    fig = Figure(; size=(400, 300))

    ax = Axis(
        fig[1, 1];
        xticks=(1:4, tick_labels),
        # xticklabelcolor=majority_colour,
        yticks=(1:4, tick_labels),
        # yticklabelcolor=minority_colour,
        xaxisposition=(:top),
        # yticklabelrotation=π/6,
        # xticklabelrotation=π/6,
        yreversed=true,
        aspect=DataAspect(),
        xlabel="Norm: in-group interactions",
        ylabel="Norm: out-group interactions",
    )
    # cmap = cgrad(cgrad(:Hiroshige; rev=true)[2:end])
    cmap = :thermal
    hmap = heatmap!(
        ax,
        Matrix{Float64}(df_famous_norms_minority[:, 2:end][[2, 4, 1, 3], [2, 4, 1, 3]]);
        colormap=cmap,
        colorrange=(0, 1),
        # tellheight=true,
        # lowclip=darkgrey",
    )
    Colorbar(
        fig[1, 2], hmap; label="Cooperativeness towards minority", height=Relative(13 / 14)
    )#, tickformat = values -> ["$(round(Int, value*100))%" for value in values])
    resize_to_layout!(fig)
    for filetype in ("png", "pdf")
        save("projects/aamas/figures/analytical_famous_minority.$filetype", fig)
    end
    fig
end

let
    df_famous_norms_majority = @chain df begin
        generate_quadrant_df(; p=p)
        # select(:norm, :majority_strat, :minority_strat, :cooperation, :fairness)
        transform(:norm => ByRow(split_norm) => AsTable)
        subset([:ingroup_norm, :outgroup_norm] .=> ByRow(in(famous_norms)))
        groupby(:norm)
        combine(sdf -> first(sort(sdf, :cooperation; rev=true)))
        # sort(:cooperation, rev=true)
        # subset(:outgroup_norm => ByRow(==(12)))
        select(:ingroup_norm, :outgroup_norm, :prr)
        unstack(:outgroup_norm, :prr)
        # select(:ingroup_norm, reverse(["12", "8", "9", "13"]))
    end

    df_famous_norms_minority = @chain df begin
        generate_quadrant_df(; p=p)
        # select(:norm, :majority_strat, :minority_strat, :cooperation, :fairness)
        transform(:norm => ByRow(split_norm) => AsTable)
        subset([:ingroup_norm, :outgroup_norm] .=> ByRow(in(famous_norms)))
        groupby(:norm)
        combine(sdf -> first(sort(sdf, :cooperation; rev=true)))
        # sort(:cooperation, rev=true)
        # subset(:outgroup_norm => ByRow(==(12)))
        select(:ingroup_norm, :outgroup_norm, :pbr)
        unstack(:outgroup_norm, :pbr)
        # select(:ingroup_norm, reverse(["12", "8", "9", "13"]))
    end

    let
        # tick_labels = ["Shunning", "SternJudging", "ImageScore", "SimpleStanding"]
        tick_labels = ["SJ", "SS", "SH", "IS"]
        cmap = :thermal
        # ["SH", "SJ", "IS", "SS"] # Original order 

        fig = Figure(; size=(600, 250))
        ax_settings = (;
            xticks=(1:4, tick_labels),
            yticks=(1:4, tick_labels),
            xaxisposition=(:top),
            yreversed=true,
            aspect=DataAspect(),
            xlabel="In-group norm",
            ylabel="Out-group norm",
        )
        hmap_settings = (colormap=cmap, colorrange=(0, 1))
        ax_majority = Axis(fig[1, 1]; ax_settings...)
        ax_minority = Axis(fig[1, 2]; ax_settings...)
        hmap_majority = heatmap!(
            ax_majority,
            Matrix{Float64}(df_famous_norms_majority[:, 2:end][[2, 4, 1, 3], [2, 4, 1, 3]]);
            hmap_settings...,
        )
        hmap_minority = heatmap!(
            ax_minority,
            Matrix{Float64}(df_famous_norms_minority[:, 2:end][[2, 4, 1, 3], [2, 4, 1, 3]]);
            hmap_settings...,
        )
        label_settings = (; font=:bold_italic, padding=(0, 0, -0, 0))
        Label(fig[1, 1, TopLeft()], "Majority\nGroup"; label_settings...)
        Label(fig[1, 2, TopLeft()], "Minority\nGroup"; label_settings...)
        Colorbar(
            fig[1:end, end + 1],
            hmap_majority;
            label="Cooperativeness\nexperienced by group",
            height=Relative(13 / 14),
            # vertical=false,
            # width=Relative(13/14),
        )#, tickformat = values -> ["$(round(Int, value*100))%" for value in values])
        resize_to_layout!(fig)
        for filetype in ("png", "pdf")
            save("projects/aamas/figures/analytical_famous_by_group.$filetype", fig)
        end
        fig
    end
end
