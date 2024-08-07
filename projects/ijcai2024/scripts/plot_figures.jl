using CairoMakie
using CSV
using DataFrames
using Format: format
using GeometryBasics
using IR
using IRUtils
using StaticArrays
using Tidier

p = (;
    maj_em=0.01,
    min_em=0.01,
    judge_em=0.01,
    maj_pm=SA[0.0, 0.0],
    min_pm=SA[0.0, 0.0],
    judge_pm=SA[0.0, 0.0, 0.0],
    prop_maj=0.9,
    utilities=SA[5, 5, 1, 1],
)

# Shapes
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

# Figure 2

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
    return Matrix{Int64}(cdf[:, 2:4])
end

let
    fairness_data = subset(find_ESS(p), :is_ess)
    tot_matrix = get_cell_counts(find_ESS(p))
    ess_matrix = get_cell_counts(fairness_data)
    pct_matrix = ess_matrix ./ tot_matrix
    mat_size = size(tot_matrix)

    tick_labels = ["Always defect", "Group-agnostic", "Discriminatory"]
    fig = Figure(; size=(600, 450))
    ax = Axis(
        fig[1, 1];
        xticks=(1:mat_size[2], tick_labels),
        yticks=(1:mat_size[1], tick_labels),
        xaxisposition=(:top),
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
    Colorbar(fig[1, 2], hmap; label="Number of stable combinations")
    for i in 1:3, j in 1:3
        ess_matrix[i, j] == 0 && continue
        txtcolor = ess_matrix[i, j] < 150 ? :white : :black
        plot_text_value = "$(ess_matrix[i, j]) of $(format(tot_matrix[i,j], commas=true))"
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
    for filetype in ("png", "pdf")
        save("projects/ijcai2024/figures/stable_nss_combinations.$filetype", fig)
    end
    fig
end

let # Figure 3 and D.2 in the appendix
    ## The shapes we use in the plot

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

    ## Data for plot
    df = find_ESS(p)

    # Data for scatter
    df_scatter = generate_quadrant_df(df; p)
    transform!(
        df_scatter,
        [:majority_strat, :minority_strat] .=>
            ByRow(categorise_strategy_old) .=>
                [:majority_strat_category, :minority_strat_category],
    )
    subset!(df_scatter, :majority_strat => ByRow(!=(0)))

    # Data for arrows
    df_arrows = let
        p_low = (; p..., utilities=SA[1.25, 1.25, 1, 1])
        fairness_data_low = generate_quadrant_df(subset(find_ESS(p_low), :is_ess); p=p_low)
        fairness_data_high = generate_quadrant_df(subset(find_ESS(p), :is_ess); p)
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
    df_arrows.fairness_diff = df_arrows.fairness_low .- df_arrows.fairness_high
    subset!(df_arrows, :fairness_diff => ByRow(!=(0)))

    # Plot settings
    ticks = 0:0.25:1
    offset = 0.025
    lims = (0 - offset, 1 + offset)
    cmap = cgrad(:Hiroshige, 3; rev=true, categorical=true)
    markersize = 7

    ## Draw figure
    fig = Figure(; size=(600, 550))
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
    limits!(ax, lims, lims)

    # We iterate over each row and plot the maj and min semi-circles for each
    # row as to superimpose the markers in the right order.
    foreach(eachrow(df_scatter)) do row
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

    # Make custom legend:
    legend = let
        marker = Polygon(decompose(Point2f, Circle(Point2f(0), 1)))
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
        Legend(
            fig[2, 1],
            [inner_outer_elements, color_elements, shape_elements],
            [inner_outer_labels, color_labels, shape_labels],
            ["Whose strategy?", "What kind of strategy?", "What kind of norm?"];
            nbanks=3,
            orientation=:horizontal,
            tellwidth=false,
            tellheight=true,
        )
    end
    display(fig)
    for filetype in ("pdf", "png")
        # save("projects/ijcai2024/figures/coop_fairness_strategy_scatter.$filetype", fig)
    end
    ## Draw arrows showing the difference in fairness between ratios
    arrows!(
        ax,
        df_arrows.cooperation_high,
        df_arrows.fairness_high,
        df_arrows.cooperation_low .- df_arrows.cooperation_high,
        df_arrows.fairness_low .- df_arrows.fairness_high,
    )
    for filetype in ("pdf", "png")
        # save("./projects/ijcai2024/figures/coop_fairness_arrows.$filetype", fig)
    end
    fig
end

# Figure 4
let
    # Generate data
    famous_norms = (8, 9, 12, 13)
    df = find_ESS(p)
    _intermediate_df = @chain df begin
        generate_quadrant_df(; p=p)
        transform(:norm => ByRow(split_norm) => AsTable)
        subset([:ingroup_norm, :outgroup_norm] .=> ByRow(in(famous_norms)))
        groupby(:norm)
        combine(sdf -> first(sort(sdf, :cooperation; rev=true)))
    end

    df_famous_norms_majority = @chain _intermediate_df begin
        select(:ingroup_norm, :outgroup_norm, :prr)
        unstack(:outgroup_norm, :prr)
    end

    df_famous_norms_minority = @chain _intermediate_df begin
        select(:ingroup_norm, :outgroup_norm, :pbr)
        unstack(:outgroup_norm, :pbr)
    end

    # Plot figure
    tick_labels = ["SJ", "SS", "SH", "IS"]
    cmap = :thermal
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
    )
    resize_to_layout!(fig)
    for filetype in ("png", "pdf")
        save("projects/ijcai2024/figures/analytical_famous_by_group.$filetype", fig)
    end
    fig
end

# Figure 5
let
    # Run `generate_data.jl` first to be able to reproduce this plot
    p_fig_5 = (;
        maj_em=0.01,
        min_em=0.01,
        judge_em=0.01,
        maj_pm=SA[0.0, 0.0],
        min_pm=SA[0.0, 0.0],
        judge_pm=SA[0.0, 0.0, 0.0],
        prop_maj=0.9,
        utilities=SA[10, 10, 1, 1],
    )
    df_rl_granular = CSV.read(
        "projects/ijcai2024/data/figure5/granular_rl_data_10.csv",
        DataFrame;
        stripwhitespace=true,
    )
    begin # Data for plot
        df = find_ESS(p)
        df_egt = generate_quadrant_df(df; p=p_fig_5)
        transform!(
            df_egt,
            [:majority_strat, :minority_strat] .=>
                ByRow(categorise_strategy_old) .=>
                    [:majority_strat_category, :minority_strat_category],
        )
        select!(df_egt, :norm, :cooperation, :fairness)
    end

    ticks = 0:0.25:1
    fig = Figure(; size=(600, 514))

    scatter_settings = (markersize=15, strokewidth=1.2)
    markers = [:circle, :rect, :cross]
    cmap = cgrad(:Egypt, 80; categorical=true)

    df_new = @chain generate_quadrant_df(df; p=p_fig_5) begin
        groupby(:norm)
        combine(sdf -> first(sort(sdf, :cooperation; rev=true)))
        subset([:majority_strat, :minority_strat] => ByRow((x, y) -> !(x == y == 0)))
        select(:norm, :cooperation, :fairness)
        leftjoin(df_rl_granular; on=:norm, makeunique=true)
        rename(:cooperation_1 => :cooperation_rl, :fairness_1 => :fairness_rl)
    end

    transform!(
        df_new, [:cooperation, :fairness] .=> ByRow(>(0.5)) .=> [:x_quadrant, :y_quadrant]
    )
    transform!(
        df_new,
        [:x_quadrant, :y_quadrant] => ByRow((a, b) -> evalpoly(2, (a, b))) => :quadrant,
    )
    df_new.coop_dir = (df_new.cooperation_rl .- df_new.cooperation) ./ 2
    df_new.fairness_dir = (df_new.fairness_rl .- df_new.fairness) ./ 2
    sdf = subset(df_new, :quadrant => ByRow(==(3)))
    df_egt = select(sdf, :norm, :cooperation, :fairness)
    df_rl2 = select(sdf, :norm, :cooperation_rl => :cooperation, :fairness_rl => :fairness)
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
    arrows!(
        ax,
        sdf.cooperation,
        sdf.fairness,
        sdf.coop_dir,
        sdf.fairness_dir;
        linewidth=1.2,
        color=(:black, 0.05),
    )
    arrows!(
        ax,
        sdf.cooperation_rl,
        sdf.fairness_rl,
        .-sdf.coop_dir,
        .-sdf.fairness_dir;
        arrowsize=0,
        linewidth=1.2,
        color=(:black, 0.05),
    )
    for (i, df) in enumerate((df_egt, df_rl2))
        d = Dict(Pair(value, i) for (i, value) in enumerate(unique(df.norm)))
        colour_vector = replace(x -> getindex(d, x), df.norm)
        scatter!(
            ax,
            df.cooperation,
            df.fairness;
            marker=markers[i],
            color=colour_vector,
            scatter_settings...,
            colormap=cmap,
            alpha=1,
        )
    end

    marker = Polygon(decompose(Point2f, Circle(Point2f(0), 1))) # <- AT THIS POINT, THE ANSWER WAS NO
    marker_elements = [
        MarkerElement(; color=:grey, marker, scatter_settings...) for marker in markers
    ]
    marker_labels = ["EGT", "RL"]
    legend = Legend(
        fig[1, 1],
        [marker_elements],
        [marker_labels],
        ["Model"];
        orientation=:horizontal,
        tellwidth=false,
        tellheight=false,
        halign=:left,
        valign=:bottom,
        margin=(70, 20, 20, 10),
    )
    resize_to_layout!(fig)
    for filetype in ("pdf", "png")
        save(
            "projects/ijcai2024/figures/granular-egt-rl-comparison-topright.$filetype", fig
        )
    end
    fig
end

# Figure 6
let
    benefits = 3:9
    seeded_agents = 0:2:50
    fig = Figure(; size=(600, 220))
    tick_labels = ["Always defect", "Group-agnostic", "Discriminatory"]
    ax = Axis(
        fig[1, 1];
        xlabel=rich(
            "Number of agents with Q-values initialised to strategy ",
            rich("\"Disc\""; font=:italic),
        ),
        ylabel="Benefit-to-cost ratio",
    )
    hmap = heatmap!(
        ax,
        seeded_agents,
        benefits,
        (x, y) -> begin
            df = CSV.read(
                "projects/ijcai2024/data/figure6/195_cooperation_$(Int(y)).csv",
                DataFrame;
                stripwhitespace=true,
            )
            subset(df, :n_seeded_agents => ByRow(==(x))).cooperation_mean[1]
        end;
        colormap=:thermal,
        colorrange=(0, 1),
    )
    Colorbar(fig[:, 2], hmap; label="Cooperation level")
    for filetype in ("pdf", "png")
        save("projects/ijcai2024/figures/heatmap_benefit_seed_cooperation.$filetype", fig)
    end
    fig
end

# Figure C.1 in the appendix (but not as pretty)
let
    let
        fig = Figure(; resolution=(600, 1100))
        cmap = :viridis
        gtop = fig[1, 1] = GridLayout()
        gmid = fig[2, 1] = GridLayout()
        gbot = fig[3, 1] = GridLayout()
        normnames = Dict(
            [192, 195, 211, 243] .=>
                ["Shunning", "SternJudging", "Norm–211", "SimpleStanding"],
        )
        # Multiple plots in one figure
        granular_output_norms = mapreduce(vcat, norms) do norm
            go = CSV.read("fig_appendix_c1/granular_data_$norm.csv")
            go.norm .= norm
            go
        end
        

        # Group by generation
        for (sdf, grid) in zip(groupby(granular_output_norms, :norm), (gtop, gmid, gbot))
            sdf_norm = only(unique(sdf.norm))
            ax_maj = Axis(
                grid[1, 1];
                ylabel="Run",
                xlabel="Strategy",
                title="Majority",
                xtickformat=x -> getindex.(Ref(strategy_names), Int.(x)),
                xticklabelrotation=π / 2,
            )
            ax_min = Axis(
                grid[1, 2];
                ylabel="Run",
                xlabel="Strategy",
                title="Minority",
                xtickformat=x -> getindex.(Ref(strategy_names), Int.(x)),
                xticklabelrotation=π / 2,
            )
            hideydecorations!(ax_min; grid=false)
            df = sort(sdf, order(:cooperation))
            df.order = 1:nrow(df)
            df_out = DataFrame(
                "order" => Int[], "group" => Bool[], (string.(0:15) .=> Ref(Float64[]))...
            )
            for row in eachrow(df)
                mip, map = row[[:minority_prevalence, :majority_prevalence]]
                for (group, mat) in enumerate((mip, map))
                    new_row = (row.order, Bool(group - 1), mean.(eachcol(mat))...)
                    push!(df_out, new_row)
                end
            end
            df_long = stack(df_out, 3:18; variable_name=:strategy, value_name=:prevalence)
            df_xticks = @chain df_long begin
                groupby([:strategy, :group])
                combine(:prevalence => maximum => :pm)
                subset(:pm => ByRow(>(0.25)))
            end

            ax_maj.xticks = parse.(Int, subset(df_xticks, :group).strategy)
            ax_min.xticks = parse.(Int, subset(df_xticks, :group => .!).strategy)
            for (group, ax) in zip((false, true), (ax_min, ax_maj))
                df_sub = subset(df_long, :group => ByRow(==(group)))
                # df.run = 1:nrow(df)
                @. df_sub.strategy = parse.(Int, df_sub.strategy)
                # @show df.strategy
                heatmap!(
                    ax,
                    df_sub.strategy,
                    df_sub.order,
                    df_sub.prevalence;
                    colormap=cmap,
                    colorrange=(0, 1),
                )
            end
            # split coop values into quintiles
            n_ticks = 5
            tick_diff_min = 4 # Ticks must be at least X apart
            # yticks, ytickformat_vec = generate_ticks(df.cooperation)
            # yticks = generate_ticks(df.cooperation; diff_min=tick_diff_min)
            yticks = generate_ticks_simple(df.cooperation; nticks=n_ticks)

            ax_coop = Axis(
                grid[1, 3];
                ylabel="Average cooperation in run",
                yaxisposition=:right,
                yticks,
                ytickformat=ys ->
                    ["$(round.(df.cooperation[Int(y)]; digits=3))" for y in ys],
            )
            hidexdecorations!(ax_coop; grid=false)
            heatmap!(
                ax_coop,
                ones(size(df.order)),
                df.order,
                df.cooperation;
                colormap=:thermal,
                colorrange=(0, 1),
            )
            for ax in (ax_maj, ax_min, ax_coop)
                hlines!(ax, yticks; color=:grey, linestyle=:dash)
            end
            Label(
                grid[0, :],
                "Prevalence of Strategies under $(normnames[sdf_norm])";
                word_wrap=true,
                font=:bold,
            )
        end
        n_plots = length(norms)
        cb = Colorbar(
            fig[n_plots + 1, :];
            limits=(0, 1),
            colormap=cmap,
            vertical=false,
            label="Strategy prevalence within population",
            flipaxis=false,
        )
        cb.tellwidth = true

        colsize!(gtop, 3, Auto(0.07))
        colsize!(gmid, 3, Auto(0.07))
        colsize!(gbot, 3, Auto(0.07))
        resize_to_layout!(fig)
        for filetype in ("png", "pdf")
            save(
                # "projects/aamas/figures/prevalence-heatmap/prevalence_heatmap_norms_$norms.$filetype",
                fig,
            )
        end
        display(fig)
    end

    # # Strategies to highlight
    # theoretical_df = generate_quadrant_df(find_ESS(p); p)
    # theoretical_df = @chain theoretical_df begin
    #     subset(
    #         :norm => ByRow(==(norm)),
    #         [:majority_strat, :minority_strat] => ByRow((x, y) -> !(x == y == 0)),
    #     )
    # end
end

# Figure D.1 in the appendix
function get_regime(p)
    df = find_ESS(p)
    df_ess = subset(df, :is_ess)
    return nrow(df_ess)
end

let
    fig = Figure(; size=(600, 450))
    tick_labels = ["Always defect", "Group-agnostic", "Discriminatory"]
    ax = Axis(fig[1, 1]; xlabel="Majority error rate", ylabel="Majority benefit")
    cmap = cgrad(cgrad(:Hiroshige; rev=true)[1:end]) # PuBu
    hmap = heatmap!(
        ax,
        0.01:0.01:0.5,
        1.0:0.01:13,
        (x, y) -> begin
            println((x, y)) # Takes a while, nice to know the progress.
            get_regime((; p..., maj_em=x, utilities=SA[y, 2, 1, 1]))
        end;
        colormap=cmap,
        lowclip=:lightgrey,
    )
    Colorbar(fig[1, 2], hmap; label="Number of stable combinations")
    for filetype in ("pdf", "png")
        save(
            "projects/ijcai2024/figures/number_of_ess_combinations_detailed.$filetype", fig
        )
    end
    fig
end


function generate_quadrant_df_nosub(_df; p)
    df = deepcopy(_df)
    info_df = DataFrame(
        [
            :majority_rep,
            :minority_rep,
            :majority_payoff,
            :minority_payoff,
            :fairness,
            :cooperation,
            :prr,
            :pbr,
            :prd,
            :pbd,
        ] .=> Ref(Float64[]),
    )
    quadrants = Int8[]
    foreach(eachrow(df)) do row
        n, r, b, _ = row
        judge, majority, minority = get_agents(n, r, b; p)
        majority_rep, minority_rep = stationary_incumbent_reputations(
            judge, majority, minority, p.prop_maj
        )
        majority_payoff, minority_payoff = incumbent_payoffs(
            majority, minority, majority_rep, minority_rep, p.prop_maj, p.utilities
        )
        prr = p_receives(majority, minority, majority_rep, p.prop_maj)
        prd = p_donates(majority, majority_rep, minority_rep, p.prop_maj)
        pbr = p_receives(minority, majority, minority_rep, 1 - p.prop_maj)
        pbd = p_donates(minority, minority_rep, majority_rep, 1 - p.prop_maj)
        fairness = let
            lower, higher = minmax(majority_payoff, minority_payoff)
            # if (n == 242 && r == 13 & b == 0)
            #     @show lower higher
            #     println(lower/higher)
            # end
            lower / higher
        end
        cooperation = p.prop_maj * prd + (1 - p.prop_maj) * pbd
        push!(
            info_df,
            (
                majority_rep,
                minority_rep,
                majority_payoff,
                minority_payoff,
                fairness,
                cooperation,
                prr,
                pbr,
                prd,
                pbd,
            ),
        )
        push!(quadrants, (cooperation < 0.5) + 2(fairness < 0.5))
    end
    df = hcat(df, info_df, quadrants)
    rename!(df, :x1 => :quadrant)
    # unique!(df, [:majority_payoff, :minority_payoff, :majority_rep, :minority_rep])
    return df
end

let # Figure 3 and D.2 in the appendix
    # Data for plots
    df_lines = DataFrame(
        :norm => Int64[],
        :majority_strat => Int64[],
        :minority_strat => Int64[],
        :prop_maj => Float64[],
        :fairness => Float64[],
        :cooperation => Float64[],
        :is_ess => Bool[]
    )
    proportion_itr = 0.52:0.005:0.9
    for prop_maj in proportion_itr
        # @show p
        @show prop_maj
        p_loop = (; p..., prop_maj)
        df = find_ESS(p_loop)
        subset!(df, :majority_strat => ByRow(!=(0)))
        qdf = generate_quadrant_df_nosub(df; p=p_loop)
        qdf[!, :prop_maj] .= prop_maj
        select!(
            qdf,
            [:norm, :majority_strat, :minority_strat, :prop_maj, :fairness, :cooperation, :is_ess],
        )
        df_lines = vcat(df_lines, qdf)
    end
    transform!(
        df_lines,
        [:majority_strat, :minority_strat] .=>
            ByRow(categorise_strategy_old) .=>
                [:majority_strat_category, :minority_strat_category],
    )
    gdf = groupby(df_lines, [:norm, :majority_strat, :minority_strat])

    # Plot settings
    ticks = 0:0.25:1
    offset = 0.025
    lims = (0 - offset, 1 + offset)
    cmap = cgrad(:Hiroshige, 3; rev=true, categorical=true)
    markersize = 5

    ## Draw figure
    fig = Figure(; size=(600, 950))
    ax1 = Axis(
        fig[1, 1];
        xlabel="Cooperativeness",
        ylabel="Fairness",
        xticks=ticks,
        yticks=ticks,
        title="",
        titlealign=:left,
        aspect=DataAspect(),
    )
    limits!(ax1, lims, lims)
    ax2 = Axis(
        fig[2, 1];
        xlabel="Cooperativeness",
        ylabel="Fairness",
        xticks=ticks,
        yticks=ticks,
        title="",
        titlealign=:left,
        aspect=DataAspect(),
    )
    limits!(ax2, lims, lims)

    # display(gdf)
    for sdf in gdf
        if !any(sdf[!, :is_ess])
            continue
        end
        if sdf[end, :is_ess]
           curr_ax = ax1 
        else
            curr_ax = ax2
        end
        lines!(curr_ax, sdf.cooperation[sdf.is_ess], sdf.fairness[sdf.is_ess]; color=:black, linewidth=0.8)
        # lines!(curr_ax, sdf.cooperation[.~sdf.is_ess], sdf.fairness[.~sdf.is_ess]; color=:black, linestyle=:dash, linewidth=0.8)
        if !is_fair(sdf.norm[1])
            shapes = (tl_semicircle, br_semicircle)
        else
            shapes = (tl_triangle, br_triangle)
        end
        for (shape, colour_column) in
            zip(shapes, (:majority_strat_category, :minority_strat_category))
            scatter!(
                curr_ax,
                sdf.cooperation[sdf.is_ess][end],
                sdf.fairness[sdf.is_ess][end];
                color=sdf[end, colour_column],
                marker=shape,
                markersize,
                colormap=cmap,
                colorrange=(0, 2),
                strokewidth=1,
                label="$colour_column",
            )
        end
    end

    # Make custom legend:
    legend = let
        marker = Polygon(decompose(Point2f, Circle(Point2f(0), 1)))
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
        Legend(
            fig[end+1, 1],
            [inner_outer_elements, color_elements, shape_elements],
            [inner_outer_labels, color_labels, shape_labels],
            ["Whose strategy?", "What kind of strategy?", "What kind of norm?"];
            nbanks=3,
            orientation=:horizontal,
            tellwidth=false,
            tellheight=true,
        )
    end
    label_settings = (; font=:bold_italic, padding=(0, 0, 0, 0), fontsize=36)
    Label(fig[1, 1, Left()], "A"; label_settings...)
    Label(fig[2, 1, Left()], "B"; label_settings...)
    for filetype in ("pdf", "png")
        save("./projects/ijcai2024/figures/coop_fairness_groupsize_trajectory.$filetype", fig)
    end
    fig
end

