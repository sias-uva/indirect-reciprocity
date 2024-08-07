fairness_data_low = generate_quadrant_df(subset(df, :is_ess); p)
fairness_data_high = generate_quadrant_df(subset(df, :is_ess); p)

joined_df = let
    high = select(
        fairness_data_high,
        [:norm, :majority_strat, :minority_strat, :cooperation, :fairness],
    )
    low = select(
        fairness_data_low,
        [:norm, :majority_strat, :minority_strat, :cooperation, :fairness],
    )
    leftjoin(
        high,
        low;
        on=[:norm, :majority_strat, :minority_strat],
        renamecols=(:_high => :_low),
    )
end

let
    # The shapes we use in the plot

    ## Inner circle vs outer circle:
    # p_big = decompose(Point2f, Circle(Point2f(0), 1))
    # p_small = decompose(Point2f, Circle(Point2f(0), 0.5))
    # outer = Polygon(p_big, [p_small])
    # inner = Polygon(p_small)

    ## Two halves of a circle, rotated 45 degrees
    # This code is incredibly finnicky due to floating-point error. Change with
    # caution!
    tl_semicircle = BezierPath([
        MoveTo(Point(-1 / √2, -1 / √2)),
        EllipticalArc(-1 / √2, -1 / √2, 1 / √2, 1 / √2, 1, 1, π, false, false),
        ClosePath(),
    ])

    br_semicircle = BezierPath([
        MoveTo(Point(1 / √2, 1 / √2)),
        EllipticalArc(1 / √2, 1 / √2, -1 / √2, -1 / √2, 1, 1, π, false, false),
        ClosePath(),
    ])

    br_triangle = BezierPath([
        MoveTo(Point(0, 0)),
        LineTo(Point(1, 1)),
        LineTo(Point(1, -1)),
        LineTo(Point(-1, -1)),
        ClosePath(),
    ])

    tl_triangle = BezierPath([
        MoveTo(Point(0, 0)),
        LineTo(Point(-1, -1)),
        LineTo(Point(-1, 1)),
        LineTo(Point(1, 1)),
        ClosePath(),
    ])

    mycircle = BezierPath([
        MoveTo(Point(1 / √2, 1 / √2)),
        EllipticalArc(1 / √2, 1 / √2, -1 / √2, -1 / √2, 1, 1, π, false, false),
        MoveTo(Point(-1 / √2, -1 / √2)),
        EllipticalArc(-1 / √2, -1 / √2, 1 / √2, 1 / √2, 1, 1, π, false, false),
        ClosePath(),
    ])

    mysquare = BezierPath([
        MoveTo(Point(0, 0)),
        LineTo(Point(-1, -1)),
        LineTo(Point(-1, 1)),
        LineTo(Point(1, 1)),
        MoveTo(Point(0, 0)),
        LineTo(Point(1, 1)),
        LineTo(Point(1, -1)),
        LineTo(Point(-1, -1)),
        ClosePath(),
    ])

    begin # Plot globals
        markersize = 8
    end
    df_plot = deepcopy(joined_df)

    transform!(
        df_plot,
        [:majority_strat, :minority_strat] .=>
            ByRow(categorise_strategy_old) .=>
                [:majority_strat_category, :minority_strat_category],
    )
    subset!(df_plot, :majority_strat => ByRow(!=(0)))

    ticks = 0:0.25:1
    fig = Figure(; resolution=(600, 700))
    ax = Axis(
        fig[1, 1];
        xlabel="Cooperativeness",
        ylabel="Fairness",
        xticks=ticks,
        yticks=ticks,
        title="",
        titlealign=:left,
        aspect=DataAspect(),
    )
    offset = 0.025
    lims = (0 - offset, 1 + offset)
    limits!(ax, lims, lims)
    cmap = cgrad(:Hiroshige, 3; rev=true, categorical=true)
    # hlines!(ax, 0.5; color=:black, linestyle=:dash)
    # vlines!(ax, 0.5; color=:black, linestyle=:dash)
    # We iterate over each row and plot the maj and min semi-circles for each
    # row as to superimpose the markers in the right order.
    foreach(eachrow(df_plot)) do row
        if !is_fair(row.norm)
            shapes = (tl_semicircle, br_semicircle)
        else
            shapes = (tl_triangle, br_triangle)
        end
        for (shape, colour_column) in
            zip(shapes, (:majority_strat_category, :minority_strat_category))
            scatter!(
                ax,
                row.cooperation_high,
                row.fairness_high;
                color=row[colour_column],
                marker=shape,
                markersize,
                colormap=cmap,
                colorrange=(0, 2),
                strokewidth=1,
                label="$colour_column",
            )
        end
    end
    df_plot.fairness_diff = df_plot.fairness_low .- df_plot.fairness_high
    df_plot_diff = subset(df_plot, :fairness_diff => ByRow(!=(0)))
    arrows!(
        ax,
        df_plot_diff.cooperation_high,
        df_plot_diff.fairness_high,
        df_plot_diff.cooperation_low .- df_plot_diff.cooperation_high,
        df_plot_diff.fairness_low .- df_plot_diff.fairness_high,
    )
    ## For debugging purposes, numbers should align with comments from `categorise_strategy_old`
    # cb = Colorbar(fig[1, 2]; colormap=cmap, nsteps=2, label="Is ESS?", ticks=0:2, colorrange=(0, 2),)
    x_offset = 0.25
    x_offsets = (x_offset, -x_offset, x_offset, -x_offset)
    y_offset = 0.1
    y_offsets = (y_offset, y_offset, -y_offset, -y_offset)
    rowsize!(fig.layout, 1, Aspect(1, 1))

    # IS IT REALLY WORTH IT?
    # WAS YOUR SANITY REALLY WORTH IT?

    # Make custom legend:
    marker = Polygon(decompose(Point2f, Circle(Point2f(0), 1))) # <- AT THIS POINT, THE ANSWER WAS NO
    inner_outer_elements = [
        MarkerElement(; color=:darkgrey, marker, markersize, strokewidth=1) for
        marker in (tl_semicircle, br_semicircle)
    ]
    color_elements = let
        vect = [
            MarkerElement(; color, marker, markersize, strokewidth=1) for
            color in getindex.(Ref(cmap), 1:3)
        ]
        vect[1], vect[2] = vect[2], vect[1]
        vect
    end
    shape_elements = [
        MarkerElement(; color=:darkgrey, marker, markersize, strokewidth=1) for
        marker in (mycircle, mysquare)
    ]
    inner_outer_labels = ["Majority strategy", "Minority strategy"]
    color_labels = ["Always defect", "Group-agnostic", "Discriminatory"]
    shape_labels = ["Unfair", "Fair"]
    legend = Legend(
        fig[2, 1],
        [inner_outer_elements, color_elements, shape_elements],
        [inner_outer_labels, color_labels, shape_labels],
        ["Whose strategy?", "What kind of strategy?", "What kind of norm?"];
        nbanks=3,
        orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
    )

    for filetype in ("pdf", "png")
        save("./projects/aamas/figures/coop_fairness_strategy_scatter_diff.$filetype", fig)
    end
    fig
end

joined_df.fairness_diff = joined_df.fairness_low .- joined_df.fairness_high
joined_df.abs_fairness_diff = abs.(joined_df.fairness_diff)

sort(
    subset(joined_df, :fairness_diff => ByRow(!isnan), :fairness_high => ByRow(>(0))),
    :abs_fairness_diff,
)

# The reputations stay exactly the same
findmax(fairness_data_high.minority_rep .- fairness_data_low.minority_rep)
findmax(fairness_data_high.minority_payoff .- fairness_data_low.minority_payoff)
for colname in names(fairness_data_high)[5:end]
    diff, idx = findmax(
        abs.(fairness_data_high[:, colname] .- fairness_data_low[:, colname])
    )
    n, ma, mi = fairness_data_high[idx, [:norm, :majority_strat, :minority_strat]]
    println("Column $colname, maxdiff: $diff at $((n, ma, mi))")
end

subset(fairness_data_high, :norm => ByRow(==(146)))
subset(fairness_data_low, :norm => ByRow(==(146)))
