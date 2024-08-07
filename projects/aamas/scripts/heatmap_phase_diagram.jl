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
);

filters = [
# [:majority_strat, :minority_strat] => ByRow((r, b) -> is_fair(r) && (b == 0 || !is_fair(b)))
];

# Norm colours by what do they discriminate vs (none, one , other, both),

df = find_ESS(p);

@chain df begin
    subset(:is_ess)
    transform(
        [:majority_strat, :minority_strat] .=>
            ByRow(categorise_strategy) .=> [:maj_cat, :min_cat],
    )
    subset(
        [:maj_cat, :min_cat] =>
            ByRow((ma, mi) -> (ma == 0 && mi == 1) || (ma == 1 && mi == 0)),
    )
    subset([:majority_strat, :minority_strat] => ByRow((ma, mi) -> (ma == 2 && mi == 0)))
end

function get_cell_counts(_df)
    category_names = Dict(1 => "AllD", 0 => "Group-agnostic", 2 => "Discriminatory")
    category_order = ["AllD", "Group-agnostic", "Discriminatory"]

    df = deepcopy(_df)
    df.category_pairs = map(eachrow(df)) do row
        majority_category = categorise_strategy_old(row.majority_strat)
        minority_category = categorise_strategy_old(row.minority_strat)
        category_names[majority_category], category_names[minority_category]
    end

    gdf = groupby(df, :category_pairs; sort=true)
    cdf = combine(gdf, nrow)
    transform!(
        cdf,
        :category_pairs .=> [ByRow(first), ByRow(last)] .=> [:majority_cat, :minority_cat],
    )
    select!(cdf, Not(:category_pairs))
    cdf = unstack(cdf, :minority_cat, :nrow; fill=0)
    cdf = cdf[indexin(category_order, cdf.majority_cat), :]
    select!(cdf, [:majority_cat, Symbol.(category_order)...])
    wide_df = cdf
    return Matrix{Int64}(wide_df[:, 2:4])
end

function get_regime(p)
    df = find_ESS(p)
    df_ess = subset(df, :is_ess)
    return nrow(df_ess)
    # return get_cell_counts(df_ess)
end

function utils_to_regime(x, y, p)
    _p = (; p..., utilities=[x, y, 1, 1])
    return get_regime(_p)
end

function error_rates_to_regime(x, y, p)
    _p = (; p..., maj_em=x, min_em=y)
    return get_regime(_p)
end

begin
    fig = Figure(; size=(600, 450))
    tick_labels = ["Always defect", "Group-agnostic", "Discriminatory"]
    ax = Axis(fig[1, 1]; xlabel="Majority error rate", ylabel="Majority benefit")
    cmap = cgrad(cgrad(:Hiroshige; rev=true)[1:end]) # PuBu
    hmap = heatmap!(
        ax,
        0.01:0.1:0.5,
        1.0:1:13,
        (x, y) -> begin
            println((x, y))
            get_regime((; p..., maj_em=x, utilities=[y, 2, 1, 1]))
        end;
        colormap=cmap,
        lowclip=:lightgrey,
    )
    Colorbar(fig[1, 2], hmap; label="Number of stable combinations")
    fig
end

# from multirun-sorted-heatmap.jl
function get_prevalence(
    norm,
    judge_characteristics,
    agent_characteristics,
    utilities,
    global_simulation_variables,
    n_runs=50,
)
    df = get_granular_data(
        norm,
        judge_characteristics,
        agent_characteristics,
        utilities,
        global_simulation_variables;
        n_runs,
    )
    prevalence = map(eachrow(df)) do row
        # warning!! hardcoded population size and majority ratio
        overall_prevalence =
            (
                mean.(eachcol(row.majority_prevalence)) * 45 +
                mean.(eachcol(row.minority_prevalence)) * 5
            ) / 50
        overall_prevalence[4] + overall_prevalence[13]
    end
    return mean(prevalence)
end

begin
    fig = Figure(; size=(600, 450))
    tick_labels = ["Always defect", "Group-agnostic", "Discriminatory"]
    ax = Axis(fig[1, 1]; xlabel="Majority error rate", ylabel="Majority benefit")
    # cmap = cgrad(cgrad(:Hiroshige; rev=true)[1:end]) # PuBu
    hmap = heatmap!(
        ax,
        0.01:0.1:0.3,
        1.0:1:15,
        (x, y) -> begin
            println((x, y))
            get_prevalence(
                195,
                judge_characteristics,
                (; agent_characteristics..., majority_ε=x),
                (; utilities..., majority_benefit=y),
                global_simulation_variables,
                10,
            )
            # get_regime((; p..., maj_em=x, utilities=[y, 2, 1, 1]))
        end;
        # colormap=cmap,
        # colorrange=(257, 400),
        lowclip=:darkgrey,
    )
    Colorbar(fig[1, 2], hmap; label="Number of stable combinations")#, tickformat = values -> ["$(round(Int, value*100))%" for value in values])
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
    subset!(df_plot, filters...)

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

    # for key in quadrant_dict_keys
    #     x_offset, y_offset = quadrant_offsets_dict[key]
    #     quadrant_label = quadrant_labels_dict[key]
    #     text!(
    #         ax,
    #         quadrant_label;
    #         position=(0.5 + x_offset, 0.5 + y_offset),
    #         word_wrap_with=2,
    #         align=(:center, :center),
    #     )
    # end

    resize_to_layout!(fig)
    # for filetype in ("pdf", "png")
    #     save("./projects/aamas/figures/coop_fairness_strategy_scatter.$filetype", fig)
    # end
    fig
end
