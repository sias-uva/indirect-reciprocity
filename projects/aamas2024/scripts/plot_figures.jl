using IR
using IRUtils
using StaticArrays
using DataFrames
using CairoMakie
using GeometryBasics
using ColorSchemes
using Colors
using CSV

##% First, set this variable to either "extended_abstract" or "poster"
figure_label = "extended_abstract"

##% Setup: Generate and load data
function alphascheme(c::Colorant, alpha::AbstractVector, newname::String)
    return ColorScheme([Colors.RGBA(c.r, c.g, c.b, a) for a in alpha], newname, "")
end

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

    filters = []
end

df = generate_quadrant_df(find_ESS(p); p)
df_rl_granular = CSV.read(
    "projects/aamas2024/data/rl_data.csv", DataFrame; stripwhitespace=true
)

##% Plot figure
begin
    ticks = 0:0.25:1
    fig = Figure(; size=(450, 590))
    ax = Axis(
        fig[1, 1];
        xlabel="Cooperativeness",
        ylabel="Fairness",
        xticks=ticks,
        yticks=ticks,
        title=rich(
            "Very few norms make ",
            rich("fair"; font=:bold_italic),
            " cooperation\n",
            rich("consistently"; font=:bold_italic),
            " learnable",
        ),
        titlealign=:left,
        titlesize=18,
        aspect=DataAspect(),
    )
    offset = 0.025
    lims = (0 - offset, 1 + offset)
    limits!(ax, lims, lims)
    norms = [195, 243, 192, 210, 209]
    norm_names = [
        "SternJudging",
        "SimpleStanding",
        "Shunning",
        "SternJudging/ImageScore",
        "Shunning/SimpleStanding",
    ]
    hiroshige = cgrad(:Hiroshige, length(norms); categorical=true)
    vals = Iterators.product(collect(0:0.01:1), collect(0:0.01:1))
    if figure_label == "poster"
        linesegments!(
            [
                (Point2(0.75, 0.75), Point2(0.7, 0.2)),
                (Point2(0.69, 0.84), Point2(0.15, 0.8)),
            ];
            color=hiroshige[4],
            linewidth=2,
            linestyle=:dash,
        )
        text!(
            0.57, 0.94; text=rich("Consistent"; color=hiroshige[1], font=:bold), rotation=0
        )
        text!(
            0.76,
            0.5;
            text=rich("Parochial"; color=hiroshige[4], font=:bold),
            rotation=-π / 40,
        )
        text!(
            0.3,
            0.83;
            text=rich("Inconsistent"; color=hiroshige[4], font=:bold),
            rotation=π / 40,
        )
        text!(
            0.36, 0.46; text=rich("Suboptimal"; color=hiroshige[5], font=:bold), rotation=0
        )
    end
    markersize = 11
    for (i, norm) in enumerate(norms)
        data = subset(df_rl_granular, :norm => ByRow(==(norm)))
        colormap = alphascheme(hiroshige[i], collect(range(0.3, 1, 256)), "hiro$i")
        scatter!(
            data.cooperation,
            data.fairness;
            color=colormap[20],
            strokewidth=1.5,
            markersize,
            strokecolor=colormap[200],
        )
    end
    marker_elements = [
        let
            colormap = alphascheme(hiroshige[i], collect(range(0.3, 1, 256)), "hiro$i")
            MarkerElement(;
                marker=:circle,
                color=colormap[20],
                strokewidth=1.5,
                markersize=15,
                strokecolor=colormap[200],
            )
        end for (i, color) in enumerate(hiroshige)
    ]
    marker_labels = norm_names
    legend = Legend(
        fig[2, 1],
        [marker_elements],
        [marker_labels],
        ["Ingroup Norm/Outgroup Norm"];
        nbanks=3,
        orientation=:horizontal,
        tellwidth=false,
        tellheight=false,
    )
    rowsize!(fig.layout, 1, Relative(0.8))
    resize_to_layout!(fig)
    for filetype in ("pdf", "png")
        save(
            "projects/aamas2024/figures/coop_fairness_rl_scatter_$figure_label.$filetype",
            fig,
        )
    end
    fig
end
