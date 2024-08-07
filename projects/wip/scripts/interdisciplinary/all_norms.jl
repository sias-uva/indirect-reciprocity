using IR
using IRUtils
using StaticArrays
using DataFrames
using CairoMakie
using GeometryBasics
using ColorSchemes
using Format: format
using Tidier

# CairoMakie.activate!()

# include("../norms.jl") # norms, simple_norms|
# include("../misc.jl")

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
);

filters = [
# [:majority_strat, :minority_strat] => ByRow((r, b) -> is_fair(r) && (b == 0 || !is_fair(b)))
];

# Norm colours by what do they discriminate vs (none, one , other, both),

df = find_ESS(p);

# @chain df begin 
#     subset(:is_ess)
#     transform([:majority_strat, :minority_strat] .=> ByRow(categorise_strategy) .=> [:maj_cat, :min_cat])
#     subset([:maj_cat, :min_cat] => ByRow((ma, mi) -> (ma==0 && mi==1) || (ma==1 && mi==0)))
#     subset([:majority_strat, :minority_strat] => ByRow((ma, mi) -> (ma==2 && mi==0)))
# end

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
    @show cdf.majority_cat
    cdf = cdf[indexin(category_order, cdf.majority_cat), :]
    select!(cdf, [:majority_cat, Symbol.(category_order)...])
    wide_df = cdf
    # wide_df = cdf |>
    #     x1 -> x1[indexin(category_order, x1.majority_cat), :] |>
    #     x2 -> select!(x2, [:majority_cat, Symbol.(category_order)...])
    # wide_df = (
    #     x1 -> (x2 -> select!(x2, [:majority_cat, Symbol.(category_order)...]))(
    #         x1[indexin(category_order, x1.majority_cat), :]
    #     )
    # )(
    #     cdf
    # )
    return Matrix{Int64}(wide_df[:, 2:4])
end

let
    fairness_data = subset(df, :is_ess)
    subset!(fairness_data, filters...)
    # majority_colour = RGBAf(0.8, 0, 0, 1.0)
    # minority_colour = RGBAf(0, 0, 0.8, 1.0)
    # display(fairness_data)
    tot_matrix = get_cell_counts(subset(df, filters...))
    ess_matrix = get_cell_counts(fairness_data)
    pct_matrix = ess_matrix ./ tot_matrix
    mat_size = size(tot_matrix)
    # @show mat_size

    println("Total possible combinations: $(sum(tot_matrix))")
    println("Total stable combinations: $(sum(ess_matrix))")
    display(ess_matrix)
    tick_labels = ["Always defect", "Group-agnostic", "Discriminatory"]
    fig = Figure(; size=(600, 450))
    ax = Axis(
        fig[1, 1];
        xticks=(1:mat_size[2], tick_labels),
        # xticklabelcolor=majority_colour,
        yticks=(1:mat_size[1], tick_labels),
        # yticklabelcolor=minority_colour,
        xaxisposition=(:top),
        # yticklabelrotation=π/6,
        # xticklabelrotation=π/6,
        yreversed=true,
        aspect=DataAspect(),
        xlabel="Majority strategy",
        ylabel="Minority strategy",
    )
    cmap = cgrad(cgrad(:Hiroshige; rev=true)[1:end]) # PuBu
    hmap = heatmap!(
        ax,
        ess_matrix;
        colormap=cmap,
        colorrange=(1, maximum(ess_matrix)),
        lowclip=:lightgrey,
    )
    Colorbar(fig[1, 2], hmap; label="Number of stable combinations")#, tickformat = values -> ["$(round(Int, value*100))%" for value in values])
    for i in 1:3, j in 1:3
        ess_matrix[i, j] == 0 && continue
        txtcolor = ess_matrix[i, j] < 150 ? :white : :black
        plot_text_value = "$(ess_matrix[i, j]) of $(Format.format(tot_matrix[i,j], commas=true))"
        plot_text_pct = round(100 * pct_matrix[i, j]; sigdigits=2)
        text!(
            ax,
            "$plot_text_value\n($plot_text_pct%)";
            position=(i, j),
            color=txtcolor,
            align=(:center, :center),
        )
    end
    resize_to_layout!(fig)
    # for filetype in ("png", "pdf")
    #     save("figures/interdisciplinary/stable_nss_combinations.$filetype", fig)
    # end
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
        markersize = 7
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
    fig = Figure(; resolution=(600, 550))
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
    # rowsize!(fig.layout, 1, Aspect(1, 1))

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
    # colsize!(fig.layout, 1, Relative(0.8))
    # resize_to_layout!(fig)
    for filetype in ("pdf", "png")
        save("./projects/aamas/figures/coop_fairness_strategy_scatter.$filetype", fig)
    end
    fig
end

begin
    qdf = generate_quadrant_df(df; p)
    subset!(qdf, [:cooperation, :fairness] .=> ByRow(>(0.75)))
    # A little buffer :))
end

(
    x -> (x -> subset(x, :cooperation => ByRow(>(0.75))))(
        sort(x, order(:cooperation; rev=true))
    )
)(
    subset(
        df, [:majority_strat, :minority_strat] .=> ByRow(==(0) ∘ categorise_strategy_old)
    ),
)

# let
#     x_offset = 0.25
#     y_offset = 0.1

#     quadrant_df = generate_quadrant_df(df; p)
#     subset!(quadrant_df, filters...)

#     quadrant_df.quadrant = map(eachrow(quadrant_df)) do row
#         (row.cooperation > 0.5), (row.fairness > 0.5)
#     end

#     quadrant_text_dict = Dict(
#         (1, 1) => "Equality\nand efficiency",
#         (0, 1) => "Low inequality,\nlow payoffs",
#         (1, 0) => "Inequality\ndespite efficiency",
#         (0, 0) => "High inequality,\n low payoffs",
#     )
#     quadrant_dict_keys = Iterators.product(false:true, false:true)
#     quadrant_count_dict = Dict{Tuple{Bool,Bool},Int}()
#     quadrant_labels_dict = Dict{Tuple{Bool,Bool},String}()
#     quadrant_offsets_dict = Dict{Tuple{Bool,Bool},Tuple{Float64,Float64}}()

#     foreach(quadrant_dict_keys) do key
#         quadrant_count_dict[key] = nrow(subset(quadrant_df, :quadrant => ByRow(==(key))))
#         quadrant_labels_dict[key] = "$(quadrant_text_dict[key]) \n(n=$(quadrant_count_dict[key]))"
#         quadrant_offsets_dict[key] = (
#             x_offset * (key[1] ? 1 : -1), y_offset * (key[2] ? 1 : -1)
#         )
#         display(subset(quadrant_df, :quadrant => ByRow(==(key))))
#     end

#     println("Total stable combinations: $(nrow(quadrant_df))")

#     ticks = 0:0.25:1
#     resolution = (550, 500)
#     fig = Figure(; resolution)
#     ax = Axis(
#         fig[1, 1];
#         xlabel="Cooperativeness of Society",
#         ylabel="Fairness of Society",
#         xticks=ticks,
#         yticks=ticks,
#         # title="The distribution of stable strategies in terms of\n cooperativeness and fairness",
#         titlesize=24.0f0,
#         titlealign=:left,
#         aspect=DataAspect(),
#     )
#     offset = 0.025
#     lims = (0 - offset, 1 + offset)
#     limits!(ax, lims, lims)
#     cmap = cgrad(:Hiroshige; rev=true)
#     hlines!(ax, 0.5; color=:black, linestyle=:dash)
#     vlines!(ax, 0.5; color=:black, linestyle=:dash)
#     sc = scatter!(
#         ax,
#         quadrant_df.cooperation,
#         quadrant_df.fairness;
#         colormap=cmap,
#         color=quadrant_df.minority_payoff,
#         markersize=17,
#         strokewidth=1,
#     )
#     Colorbar(fig[1, 2], sc; label="Minority payoff")

#     for key in quadrant_dict_keys
#         x_offset, y_offset = quadrant_offsets_dict[key]
#         quadrant_label = quadrant_labels_dict[key]
#         text!(
#             ax,
#             quadrant_label;
#             position=(0.5 + x_offset, 0.5 + y_offset),
#             word_wrap_with=2,
#             align=(:center, :center),
#         )
#     end
#     rowsize!(fig.layout, 1, Aspect(1, 1))
#     save("figures/interdisciplinary/coop_fairness_payoff_scatter.pdf", fig)
#     fig
# end

# let
#     quadrant_df = generate_quadrant_df(df; p)
#     subset!(quadrant_df, filters...)
#     quadrant_size = map(0:3) do quadrant_i
#         sdf = subset(quadrant_df, :quadrant => ByRow(==(quadrant_i)))
#         size(sdf, 1)
#     end
#     disc_rel_labels = Dict{Bool,String}(false => "", true => "Disc relation")
#     disc_rep_labels = Dict{Bool,String}(false => "", true => "Disc reputation")
#     discrimination_dict = Dict{Int64,String}(
#         1 => "No discrimination",
#         2 => "Discriminates on relation",
#         3 => "Discriminates on reputation",
#         4 => "Discriminates on reputation and relation",
#     )
#     # Norm colours by what do they discriminate vs (none, one , other, both),
#     quadrant_df.discrimination = map(eachrow(quadrant_df)) do row
#         n = iNorm(row.norm)
#         # Does the norm discriminate based on
#         disc_rel = n[1, :, :] != n[2, :, :] # Recipient group relation
#         disc_rep = n[:, 1, :] != n[:, 2, :] # Recipient reputation
#         1 + (1 * disc_rel) + (2 * disc_rep)
#     end
#     quadrant_df.label = map(eachrow(quadrant_df)) do row
#         "$(discrimination_dict[row.discrimination])"
#     end
#     ticks = 0:0.25:1
#     fig = Figure(; resolution=(500, 500))
#     ax = Axis(
#         fig[1, 1];
#         xlabel="Cooperativeness",
#         ylabel="Fairness",
#         xticks=ticks,
#         yticks=ticks,
#         title="The distribution of stable strategies in terms of\n cooperativeness and fairness",
#         titlealign=:left,
#         aspect=DataAspect(),
#     )
#     offset = 0.025
#     lims = (0 - offset, 1 + offset)
#     limits!(ax, lims, lims)
#     cmap = cgrad(:Hiroshige; rev=true)
#     hlines!(ax, 0.5; color=:black, linestyle=:dash)
#     vlines!(ax, 0.5; color=:black, linestyle=:dash)
#     foreach(eachrow(quadrant_df)) do row
#         scatter!(
#             ax,
#             row.cooperation,
#             row.fairness;
#             colormap=:Hiroshige,
#             colorrange=(1, 4),
#             color=row.discrimination,
#             label=row.label,
#             strokewidth=1,
#         )
#     end
#     # Colorbar(fig[1, 2], sc; label="Minority payoff")
#     # quadrant_text = (
#     #     "Everyone wins", "Most lose,\nsome win", "Most win,\nsome lose", "Few win"
#     # )
#     # quadrant_labels = [text * "\n(n=$n)" for (text, n) in zip(quadrant_text, quadrant_size)]
#     x_offset = 0.25
#     x_offsets = (x_offset, -x_offset, x_offset, -x_offset)
#     y_offset = 0.1
#     y_offsets = (y_offset, y_offset, -y_offset, -y_offset)
#     # for (quadrant_label, x_offset, y_offset) in zip(quadrant_labels, x_offsets, y_offsets)
#     #     text!(
#     #         ax,
#     #         quadrant_label;
#     #         position=(0.5 + x_offset, 0.5 + y_offset),
#     #         word_wrap_with=2,
#     #         align=(:center, :center)
#     #     )
#     # end

#     rowsize!(fig.layout, 1, Aspect(1, 1))
#     axislegend("How does the norm discriminate?"; position=:lc, merge=true)
#     save("./figures/interdisciplinary/coop_fairness_discrimination_scatter.pdf", fig)
#     fig
# end

# let
#     quadrant_df = generate_quadrant_df(df; p)
#     subset!(quadrant_df, filters...)
#     quadrant_size = map(0:3) do quadrant_i
#         sdf = subset(quadrant_df, :quadrant => ByRow(==(quadrant_i)))
#         size(sdf, 1)
#     end
#     disc_rel_labels = Dict{Bool,String}(false => "", true => "Disc relation")
#     disc_rep_labels = Dict{Bool,String}(false => "", true => "Disc reputation")
#     discrimination_dict = Dict{Int64,String}(
#         1 => "No discrimination",
#         2 => "Discriminates on relation",
#         3 => "Discriminates on reputation",
#         4 => "Discriminates on reputation and relation",
#     )
#     # Norm colours by what do they discriminate vs (none, one , other, both),
#     quadrant_df.discrimination = map(eachrow(quadrant_df)) do row
#         n = iNorm(row.norm)
#         # Does the norm discriminate based on
#         disc_rel = n[1, :, :] != n[2, :, :] # Recipient group relation
#         disc_rep = n[:, 1, :] != n[:, 2, :] # Recipient reputation
#         1 + (1 * disc_rel) + (2 * disc_rep)
#     end
#     quadrant_df.label = map(eachrow(quadrant_df)) do row
#         "$(discrimination_dict[row.discrimination])"
#     end
#     ticks = 0:0.25:1
#     fig = Figure(; resolution=(500, 500))
#     ax = Axis(
#         fig[1, 1];
#         xlabel="Cooperativeness",
#         ylabel="Fairness",
#         xticks=ticks,
#         yticks=ticks,
#         title="The distribution of stable strategies in terms of\n cooperativeness and fairness",
#         titlealign=:left,
#         aspect=DataAspect(),
#     )
#     offset = 0.025
#     lims = (0 - offset, 1 + offset)
#     limits!(ax, lims, lims)
#     cmap = cgrad(:Hiroshige; rev=true)
#     hlines!(ax, 0.5; color=:black, linestyle=:dash)
#     vlines!(ax, 0.5; color=:black, linestyle=:dash)
#     foreach(eachrow(quadrant_df)) do row
#         scatter!(
#             ax,
#             row.cooperation,
#             row.fairness;
#             colormap=:Hiroshige,
#             colorrange=(1, 4),
#             color=row.discrimination,
#             label=row.label,
#             strokewidth=1,
#         )
#     end
#     # Colorbar(fig[1, 2], sc; label="Minority payoff")
#     # quadrant_text = (
#     #     "Everyone wins", "Most lose,\nsome win", "Most win,\nsome lose", "Few win"
#     # )
#     # quadrant_labels = [text * "\n(n=$n)" for (text, n) in zip(quadrant_text, quadrant_size)]
#     x_offset = 0.25
#     x_offsets = (x_offset, -x_offset, x_offset, -x_offset)
#     y_offset = 0.1
#     y_offsets = (y_offset, y_offset, -y_offset, -y_offset)
#     # for (quadrant_label, x_offset, y_offset) in zip(quadrant_labels, x_offsets, y_offsets)
#     #     text!(
#     #         ax,
#     #         quadrant_label;
#     #         position=(0.5 + x_offset, 0.5 + y_offset),
#     #         word_wrap_with=2,
#     #         align=(:center, :center)
#     #     )
#     # end

#     rowsize!(fig.layout, 1, Aspect(1, 1))
#     axislegend("How does the strategy discriminate?"; position=:lc, merge=true)
#     # save("./figures/interdisciplinary/coop_fairness_strategy_scatter.pdf", fig)
#     fig
# end

# let
#     insularity_df = generate_quadrant_df(df; p)
#     subset!(insularity_df, filters...)
#     info_df = DataFrame([:majority_insularity, :minority_insularity] .=> Ref(Float64[]))
#     foreach(eachrow(insularity_df)) do row
#         norm, majority_strat, minority_strat, _, majority_rep, minority_rep = row
#         judge, majority, minority = get_agents(norm, majority_strat, minority_strat; p)
#         pRdR = lerp(SA[majority(1, 0), majority(1, 1)], majority_rep)
#         pRdB = lerp(SA[majority(0, 0), majority(0, 1)], minority_rep)
#         pBdR = lerp(SA[minority(0, 0), minority(0, 1)], majority_rep)
#         pBdB = lerp(SA[minority(1, 0), minority(1, 1)], minority_rep)
#         majority_insularity = pRdR / (pRdR + pRdB)
#         minority_insularity = pBdB / (pBdB + pBdR)
#         push!(info_df, (majority_insularity, minority_insularity))
#     end
#     insularity_df = hcat(insularity_df, info_df)
#     ticks = 0:0.25:1
#     fig = Figure(; resolution=(500, 500))
#     ax = Axis(
#         fig[1, 1];
#         xlabel="Insularity of majority group cooperation",
#         ylabel="Insularity of minority group cooperation",
#         xticks=ticks,
#         yticks=ticks,
#         title="Insularity of cooperation by group\nfor stable combinations",
#         titlealign=:left,
#         aspect=DataAspect(),
#     )
#     offset = 0.025
#     lims = (0 - offset, 1 + offset)
#     limits!(ax, lims, lims)
#     cmap = cgrad(:Hiroshige; rev=true)
#     sc = scatter!(
#         ax,
#         insularity_df.majority_insularity,
#         insularity_df.minority_insularity;
#         colormap=cmap,
#         color=insularity_df.minority_payoff,
#         strokewidth=1,
#     )

#     ## Colorbar
#     Colorbar(fig[1, 2], sc; label="Minority payoff")

#     rowsize!(fig.layout, 1, Aspect(1, 1))
#     for filetype in ("pdf", "png")
#         save("./figures/interdisciplinary/insularity_scatter.$filetype", fig)
#     end
#     fig
# end
