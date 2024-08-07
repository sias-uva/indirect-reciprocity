using IR
using StaticArrays
using DataFrames
using CairoMakie
using ColorSchemes

include("norms.jl") # norms, simple_norms
include("misc.jl")

player_execution_mistake_rate = 0.01
judge_execution_mistake_rate = 0.01
player_perception_mistake_rate = SA[0.00, 0.00]
judge_perception_mistake_rate = 0.0
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

begin
    df = find_ESS(p)
    transform!(df, :norm => ByRow(split_norm) => AsTable)
    transform!(
        df,
        [:ingroup_norm, :outgroup_norm] .=>
            ByRow(get_norm_name!) .=> [:ingroup_norm_name, :outgroup_norm_name],
    )
    # subset!(df, :is_ess)
    subset!(df, :majority_strat => ByRow(!=(0)))
end

# The shapes we use in the plot
outer = BezierPath([
    MoveTo(Point(1, 0)),
    EllipticalArc(Point(0, 0), 1, 1, 0, 0, 2pi),
    MoveTo(Point(0.5, 0)),
    EllipticalArc(Point(0, 0), 0.5, 0.5, 0, 0, -2pi),
    # MoveTo(Point(1, 0)),
    # MoveTo(Point(0.5, 0.5)),
    # LineTo(Point(0.5, -0.5)),
    # LineTo(Point(-0.5, -0.5)),
    # LineTo(Point(-0.5, 0.5)),
    ClosePath(),
])

inner = BezierPath([
    MoveTo(Point(0.5, 0)),
    EllipticalArc(Point(0, 0), 0.5, 0.5, 0, 0, 2pi),
    # MoveTo(Point(0.5, 0.5)),
    # LineTo(Point(0.5, -0.5)),
    # LineTo(Point(-0.5, -0.5)),
    # LineTo(Point(-0.5, 0.5)),
    ClosePath(),
])

# A nicer way of doing the same thing:
p_big = decompose(Point2f, Circle(Point2f(0), 1))
p_small = decompose(Point2f, Circle(Point2f(0), 0.5))
outer = Polygon(p_big, [p_small])
inner = Polygon(p_small)

begin
    begin # Plot globals
        markersize = 7
    end

    df_plot = generate_quadrant_df(df; p)# from pnas_plots
    transform!(df_plot, :norm => ByRow(split_norm) => AsTable)
    transform!(
        df_plot,
        [:ingroup_norm, :outgroup_norm] .=>
            ByRow(get_norm_name!) .=> [:ingroup_norm_name, :outgroup_norm_name],
    )
    # subset!(df_plot, :is_ess)
    subset!(df_plot, :majority_strat => ByRow(!=(0)))

    ticks = 0:0.25:1
    fig = Figure(; resolution=(700, 600))
    ax = Axis(
        fig[1, 1];
        xlabel="Cooperativeness",
        ylabel="Fairness",
        xticks=ticks,
        yticks=ticks,
        title="The distribution of stable strategies in terms of\n cooperativeness and fairness",
        titlealign=:left,
        aspect=DataAspect(),
    )
    offset = 0.025
    lims = (0 - offset, 1 + offset)
    limits!(ax, lims, lims)
    cmap = cgrad(:Hiroshige; rev=true)
    foreach(eachrow(df_plot)) do row
        doplot = !(
            row.ingroup_norm_name in keys(simple_norms) &&
            row.outgroup_norm_name in keys(simple_norms)
        )
        # doplot || return
        # println("$(row.ingroup_norm_name), $(row.outgroup_norm_name)")
        for (shape, col) in zip([inner, outer], [:ingroup_norm_name, :outgroup_norm_name])
            is_leading = row[col] in keys(simple_norms)
            scatter!(
                ax,
                row.cooperation,
                row.fairness;
                colormap=:Hiroshige,
                colorrange=(1, 2),
                color=is_leading + 1,
                marker=shape,
                strokewidth=1,
                label="$is_leading",
                markersize,
            )
        end
    end

    hlines!(ax, 0.5; color=:black, linestyle=:dash)
    vlines!(ax, 0.5; color=:black, linestyle=:dash)
    x_offset = 0.25
    x_offsets = (x_offset, -x_offset, x_offset, -x_offset)
    y_offset = 0.1
    y_offsets = (y_offset, y_offset, -y_offset, -y_offset)
    rowsize!(fig.layout, 1, Aspect(1, 1))
    # Make custom legend:
    inner_outer_elements = [
        MarkerElement(; color=:black, marker=outer),
        MarkerElement(; color=:black, marker=inner),
    ]
    color_elements = [
        MarkerElement(;
            color=get(ColorSchemes.colorschemes[:Hiroshige], 1),
            marker=:circle,
            markersize=20,
        ),
        MarkerElement(;
            color=get(ColorSchemes.colorschemes[:Hiroshige], 0),
            marker=:circle,
            markersize=20,
        ),
    ]
    inner_outer_labels = ["Outgroup norm", "Ingroup norm"]
    color_labels = ["Leading 8", "Not Leading 8"]

    legend = Legend(
        fig[1, 2],
        [inner_outer_elements, color_elements],
        [inner_outer_labels, color_labels],
        ["Which part of norm?", "Is norm \"Leading 8?\""],
    )

    #save("./figures/interdisciplinary/coop_fairness_payoff_scatter.pdf", fig)
    fig
end

# begin
#     fig = Figure()
#     ax = Axis(fig[1, 1], aspect=DataAspect())
#     scatter!(
#         ax,
#         [1, 2, 3, 4],
#         [1, 2, 3, 4],
#         marker=outer,
#         # colormap=:Hiroshige,
#         colorrange=(1, 2),
#         color=[1, 1, 2, 2],
#         strokewidth=0
#     )
#     scatter!(
#         ax,
#         [1, 2, 3, 4],
#         [1, 2, 3, 4],
#         marker=inner,
#         # colormap=:Hiroshige,
#         colorrange=(1, 2),
#         color=[1, 2, 1, 2],
#         strokewidth=0,
#     )
#     fig
# end
