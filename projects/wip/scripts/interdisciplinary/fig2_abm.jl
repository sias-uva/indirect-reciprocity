using IR
using IRUtils
using StaticArrays
using DataFrames
using CairoMakie
using GeometryBasics
using ColorSchemes
using Format
using CSV

# CairoMakie.activate!()

# include("../norms.jl") # norms, simple_norms|
# include("../misc.jl")

begin
    player_execution_mistake_rate = 0.01
    judge_execution_mistake_rate = 0.01
    player_perception_mistake_rate = SA[0.00, 0.00]
    judge_perception_mistake_rate = 0.00
    proportion_incumbents_majority = 0.9
    utilities = SA[1.1, 1.1, 1, 1]

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

# Norm colours by what do they discriminate vs (none, one , other, both),

df = generate_quadrant_df(find_ESS(p); p)
df_abm = CSV.read("projects/aamas/data/abm_data.csv", DataFrame; stripwhitespace=true)
df_rl = CSV.read("projects/aamas/data/rl_data.csv", DataFrame; stripwhitespace=true)

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
        select!(
            df_egt,
            :norm,
            # :majority_strat_category,
            # :minority_strat_category,
            :cooperation,
            :fairness,
        )
    end

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

    scatter_settings = (markersize=30, strokewidth=1.2)
    markers = [:circle, :rect, :cross]
    cmap = cgrad(:Egypt, 4; categorical=true)
    for (i, df) in enumerate((df_egt, df_abm, df_rl))
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
    marker_labels = ["EGT", "ABM", "RL"]
    color_labels = ["150", "Shunning", "Stern Judging", "Simple Standing"]
    legend = Legend(
        fig[2, 1],
        [marker_elements, color_elements],
        [marker_labels, color_labels],
        ["Model", "Norm"];
        nbanks=2,
        orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
    )
    resize_to_layout!(fig)
    for filetype in ("pdf", "png")
        save("projects/aamas/figures/egt-rl-abm-comparison.$filetype", fig)
    end
    fig
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
    inner = BezierPath([
        MoveTo(Point(-1 / √2, -1 / √2)),
        EllipticalArc(-1 / √2, -1 / √2, 1 / √2, 1 / √2, 1, 1, π, false, false),
        ClosePath(),
    ])

    outer = BezierPath([
        MoveTo(Point(1 / √2, 1 / √2)),
        EllipticalArc(1 / √2, 1 / √2, -1 / √2, -1 / √2, 1, 1, π, false, false),
        ClosePath(),
    ])

    begin # Plot globals
        markersize = 8
    end
    df_plot = generate_quadrant_df(df; p)

    transform!(
        df_plot,
        [:majority_strat, :minority_strat] .=>
            ByRow(categorise_strategy_old) .=>
                [:majority_strat_category, :minority_strat_category],
    )
    subset!(df_plot, :majority_strat => ByRow(!=(0)))
    subset!(
        df_plot,
        :norm => ByRow(in([150, 192, 195, 243])),
        :majority_strat => ByRow(in([12, 9])),
    )

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
        for (shape, colour_column) in
            zip((inner, outer), (:majority_strat_category, :minority_strat_category))
            scatter!(
                ax,
                row.cooperation,
                row.fairness;
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

    # Arrows
    # Arrow from ABM to RL
    df_arrow = select(
        df_plot,
        :norm,
        :majority_strat,
        :minority_strat,
        :majority_strat_category,
        :minority_strat_category,
    )
    leftjoin!(df_arrow, df_abm; on=:norm)
    leftjoin!(df_arrow, df_rl; on=:norm, makeunique=true)
    display(df_arrow)
    rename!(df_arrow, :cooperation => :coop_origin, :fairness => :fairness_origin)
    rename!(
        df_arrow, :cooperation_1 => :coop_destination, :fairness_1 => :fairness_destination
    )
    df_arrow.coop_direction = df_arrow.coop_destination .- df_arrow.coop_origin
    df_arrow.fairness_direction = df_arrow.fairness_destination .- df_arrow.fairness_origin
    df_arrow.coop_halfway =
        df_arrow.coop_origin .+ ((df_arrow.coop_destination .- df_arrow.coop_origin) / 2)
    df_arrow.fairness_halfway =
        df_arrow.fairness_origin .+
        ((df_arrow.fairness_destination .- df_arrow.fairness_origin) / 2)

    # arrow_options = (linewidth = 2.0, linestyle = :dash)
    arrows!(
        df_arrow.coop_origin,
        df_arrow.fairness_origin,
        df_arrow.coop_direction / 2,
        df_arrow.fairness_direction / 2;
        arrowsize=15,
        linewidth=2,
        linestyle=:dash,
        color=:minority,
    )
    arrows!(
        df_arrow.coop_halfway,
        df_arrow.fairness_halfway,
        df_arrow.coop_direction / 2,
        df_arrow.fairness_direction / 2;
        arrowsize=0,
        linewidth=2,
        linestyle=:dash,
        color=:minority,
    )
    plot_norm_names = Dict{Int,String}(
        150 => "150", 192 => " SH", 195 => " SJ", 243 => " SS"
    )
    annotations!(
        getindex.(Ref(plot_norm_names), df_arrow.norm),
        Point.(df_arrow.coop_destination .- 0.09, df_arrow.fairness_destination .- 0.02),
    )

    foreach(eachrow(df_arrow)) do row
        for (shape, colour_column) in
            zip((inner, outer), (:majority_strat_category, :minority_strat_category))
            scatter!(
                ax,
                row.coop_destination,
                row.fairness_destination;
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
        MarkerElement(; color=:black, marker, markersize, strokewidth=1) for
        marker in (inner, outer)
    ]
    color_elements = let
        vect = [
            MarkerElement(; color, marker, markersize, strokewidth=1) for
            color in getindex.(Ref(cmap), 1:3)
        ]
        vect[1], vect[2] = vect[2], vect[1]
        vect
    end
    inner_outer_labels = ["Majority strategy", "Minority strategy"]
    color_labels = ["Always defect", "Group-agnostic", "Discriminatory"]
    legend = Legend(
        fig[2, 1],
        [inner_outer_elements, color_elements],
        [inner_outer_labels, color_labels],
        ["Whose strategy?", "What kind of strategy?"];
        nbanks=2,
        orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
    )

    df_plot.quadrant = map(eachrow(df_plot)) do row
        (row.cooperation > 0.5), (row.fairness > 0.5)
    end

    quadrant_text_dict = Dict(
        (1, 1) => "Equality\nand efficiency",
        (0, 1) => "Low inequality,\nlow payoffs",
        (1, 0) => "Inequality\ndespite efficiency",
        (0, 0) => "High inequality,\n low payoffs",
    )
    quadrant_dict_keys = Iterators.product(false:true, false:true)
    quadrant_count_dict = Dict{Tuple{Bool,Bool},Int}()
    quadrant_labels_dict = Dict{Tuple{Bool,Bool},String}()
    quadrant_offsets_dict = Dict{Tuple{Bool,Bool},Tuple{Float64,Float64}}()

    foreach(quadrant_dict_keys) do key
        quadrant_count_dict[key] = nrow(subset(df_plot, :quadrant => ByRow(==(key))))
        quadrant_labels_dict[key] = "$(quadrant_text_dict[key]) \n(n=$(quadrant_count_dict[key]))"
        quadrant_offsets_dict[key] = (
            x_offset * (key[1] ? 1 : -1), y_offset * (key[2] ? 1 : -1)
        )
        # display(subset(df_plot, :quadrant => ByRow(==(key))))
    end

    resize_to_layout!(fig)
    # for filetype in ("pdf", "png")
    #     save("figures/interdisciplinary/coop_fairness_abm_arrow.$filetype", fig)
    # end
    fig
end
