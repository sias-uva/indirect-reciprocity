using IR
using StaticArrays
using DataFrames
using CairoMakie
using ColorSchemes
using Format: format

CairoMakie.activate!()

include("../norms.jl") # norms, simple_norms|
include("../misc.jl")

player_execution_mistake_rate = 0.01
judge_execution_mistake_rate = 0.01
player_perception_mistake_rate = SA[0.00, 0.00]
judge_perception_mistake_rate = 0.00
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

filters = [
    :norm => ByRow(!is_fair),
    [:majority_strat, :minority_strat] => ByRow((r, b) -> !(is_fair(r) && is_fair(b))),
]

# Norm colours by what do they discriminate vs (none, one , other, both),
# ABM: initialise at stable state, mutation randomly adopt any strategy
# Visualisations: poster

df = find_ESS(p)

let
    #TODO, color Majority in Red and Minority in Blue (RGBAf(0.0, 0.0, 0.8, 1.0))
    function get_quadrant_counts(df)
        majority_zeros = df.majority_strat .== 0
        minority_zeros = df.minority_strat .== 0

        M00 = count(@. majority_zeros & minority_zeros)
        M01 = count(@. majority_zeros & !minority_zeros)
        M10 = count(@. !majority_zeros & minority_zeros)
        M11 = count(@. !majority_zeros & !minority_zeros)
        return [M00 M01; M10 M11]
    end

    fairness_data = subset(df, :is_ess)
    subset!(fairness_data, filters...)
    majority_colour = RGBAf(0.8, 0, 0, 1.0)
    minority_colour = RGBAf(0, 0, 0.8, 1.0)

    tot_matrix = get_quadrant_counts(subset(df, filters...))
    ess_matrix = get_quadrant_counts(fairness_data)
    pct_matrix = ess_matrix ./ tot_matrix
    println("Total possible combinations: $(sum(tot_matrix))")
    println("Total stable combinations: $(sum(ess_matrix))")
    ticks = ["AllD", "Not AllD"]
    fig = Figure(; resolution=(450, 350))
    ax = Axis(
        fig[1, 1];
        xticks=(1:2, ticks),
        xticklabelcolor=majority_colour,
        yticks=(1:2, ticks),
        yticklabelcolor=minority_colour,
        xaxisposition=(:top),
        yreversed=true,
        aspect=DataAspect(),
        xlabel="Majority group strategy",
        ylabel="Minority group strategy",
        title="Count of stable N-S-S combinations",
    )
    hmap = heatmap!(ax, ess_matrix) # , colormap = :plasma
    Colorbar(fig[1, 2], hmap; label="Number of stable combinations")#, tickformat = values -> ["$(round(Int, value*100))%" for value in values])
    for i in 1:2, j in 1:2
        txtcolor = ess_matrix[i, j] < 255 ? :white : :black
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
    for filetype in ["png", "pdf"]
        save("figures/interdisciplinary/stable_nss_combinations.$filetype", fig)
    end
    fig
end

let
    x_offset = 0.25
    y_offset = 0.1

    quadrant_df = generate_quadrant_df(df; p)
    subset!(quadrant_df, filters...)

    quadrant_df.quadrant = map(eachrow(quadrant_df)) do row
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
        quadrant_count_dict[key] = nrow(subset(quadrant_df, :quadrant => ByRow(==(key))))
        quadrant_labels_dict[key] = "$(quadrant_text_dict[key]) \n(n=$(quadrant_count_dict[key]))"
        quadrant_offsets_dict[key] = (
            x_offset * (key[1] ? 1 : -1), y_offset * (key[2] ? 1 : -1)
        )
        display(subset(quadrant_df, :quadrant => ByRow(==(key))))
    end

    println("Total stable combinations: $(nrow(quadrant_df))")

    ticks = 0:0.25:1
    resolution = (550, 500)
    fig = Figure(; resolution)
    ax = Axis(
        fig[1, 1];
        xlabel="Cooperativeness of Society",
        ylabel="Fairness of Society",
        xticks=ticks,
        yticks=ticks,
        # title="The distribution of stable strategies in terms of\n cooperativeness and fairness",
        titlesize=24.0f0,
        titlealign=:left,
        aspect=DataAspect(),
    )
    offset = 0.025
    lims = (0 - offset, 1 + offset)
    limits!(ax, lims, lims)
    cmap = cgrad(:Hiroshige; rev=true)
    sc = scatter!(
        ax,
        quadrant_df.cooperation,
        quadrant_df.fairness;
        colormap=cmap,
        color=quadrant_df.minority_payoff,
        markersize=17,
        strokewidth=1,
    )
    hlines!(ax, 0.5; color=:black, linestyle=:dash)
    vlines!(ax, 0.5; color=:black, linestyle=:dash)
    Colorbar(fig[1, 2], sc; label="Minority payoff")

    for key in quadrant_dict_keys
        x_offset, y_offset = quadrant_offsets_dict[key]
        quadrant_label = quadrant_labels_dict[key]
        text!(
            ax,
            quadrant_label;
            position=(0.5 + x_offset, 0.5 + y_offset),
            word_wrap_with=2,
            align=(:center, :center),
        )
    end
    rowsize!(fig.layout, 1, Aspect(1, 1))
    save("figures/interdisciplinary/coop_fairness_payoff_scatter.pdf", fig)
    fig
end

let
    quadrant_df = generate_quadrant_df(df; p)
    subset!(quadrant_df, filters...)
    quadrant_size = map(0:3) do quadrant_i
        sdf = subset(quadrant_df, :quadrant => ByRow(==(quadrant_i)))
        size(sdf, 1)
    end
    disc_rel_labels = Dict{Bool,String}(false => "", true => "Disc relation")
    disc_rep_labels = Dict{Bool,String}(false => "", true => "Disc reputation")
    discrimination_dict = Dict{Int64,String}(
        1 => "No discrimination",
        2 => "Discriminates on relation",
        3 => "Discriminates on reputation",
        4 => "Discriminates on reputation and relation",
    )
    # Norm colours by what do they discriminate vs (none, one , other, both),
    quadrant_df.discrimination = map(eachrow(quadrant_df)) do row
        n = iNorm(row.norm)
        # Does the norm discriminate based on
        disc_rel = n[1, :, :] != n[2, :, :] # Recipient group relation
        disc_rep = n[:, 1, :] != n[:, 2, :] # Recipient reputation
        1 + (1 * disc_rel) + (2 * disc_rep)
    end
    quadrant_df.label = map(eachrow(quadrant_df)) do row
        "$(discrimination_dict[row.discrimination])"
    end
    ticks = 0:0.25:1
    fig = Figure(; resolution=(500, 500))
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
    foreach(eachrow(quadrant_df)) do row
        scatter!(
            ax,
            row.cooperation,
            row.fairness;
            colormap=:Hiroshige,
            colorrange=(1, 4),
            color=row.discrimination,
            label=row.label,
            strokewidth=1,
        )
    end
    hlines!(ax, 0.5; color=:black, linestyle=:dash)
    vlines!(ax, 0.5; color=:black, linestyle=:dash)
    # Colorbar(fig[1, 2], sc; label="Minority payoff")
    # quadrant_text = (
    #     "Everyone wins", "Most lose,\nsome win", "Most win,\nsome lose", "Few win"
    # )
    # quadrant_labels = [text * "\n(n=$n)" for (text, n) in zip(quadrant_text, quadrant_size)]
    x_offset = 0.25
    x_offsets = (x_offset, -x_offset, x_offset, -x_offset)
    y_offset = 0.1
    y_offsets = (y_offset, y_offset, -y_offset, -y_offset)
    # for (quadrant_label, x_offset, y_offset) in zip(quadrant_labels, x_offsets, y_offsets)
    #     text!(
    #         ax,
    #         quadrant_label;
    #         position=(0.5 + x_offset, 0.5 + y_offset),
    #         word_wrap_with=2,
    #         align=(:center, :center)
    #     )
    # end

    rowsize!(fig.layout, 1, Aspect(1, 1))
    axislegend("How does the norm discriminate?"; position=:lc, merge=true)
    save("./figures/interdisciplinary/coop_fairness_discrimination_scatter.pdf", fig)
    fig
end

let
    insularity_df = generate_quadrant_df(df; p)
    subset!(insularity_df, filters...)
    info_df = DataFrame([:majority_insularity, :minority_insularity] .=> Ref(Float64[]))
    foreach(eachrow(insularity_df)) do row
        norm, majority_strat, minority_strat, _, majority_rep, minority_rep = row
        judge, majority, minority = get_agents(norm, majority_strat, minority_strat; p)
        pRdR = lerp(SA[majority(1, 0), majority(1, 1)], majority_rep)
        pRdB = lerp(SA[majority(0, 0), majority(0, 1)], minority_rep)
        pBdR = lerp(SA[minority(0, 0), minority(0, 1)], majority_rep)
        pBdB = lerp(SA[minority(1, 0), minority(1, 1)], minority_rep)
        majority_insularity = pRdR / (pRdR + pRdB)
        minority_insularity = pBdB / (pBdB + pBdR)
        push!(info_df, (majority_insularity, minority_insularity))
    end
    insularity_df = hcat(insularity_df, info_df)
    ticks = 0:0.25:1
    fig = Figure(; resolution=(500, 500))
    ax = Axis(
        fig[1, 1];
        xlabel="Insularity of majority group cooperation",
        ylabel="Insularity of minority group cooperation",
        xticks=ticks,
        yticks=ticks,
        title="Insularity of cooperation by group\nfor stable combinations",
        titlealign=:left,
        aspect=DataAspect(),
    )
    offset = 0.025
    lims = (0 - offset, 1 + offset)
    limits!(ax, lims, lims)
    cmap = cgrad(:Hiroshige; rev=true)
    sc = scatter!(
        ax,
        insularity_df.majority_insularity,
        insularity_df.minority_insularity;
        colormap=cmap,
        color=insularity_df.minority_payoff,
        strokewidth=1,
    )

    ## Colorbar
    Colorbar(fig[1, 2], sc; label="Minority payoff")

    rowsize!(fig.layout, 1, Aspect(1, 1))
    for filetype in ("pdf", "png")
        save("./figures/interdisciplinary/insularity_scatter.$filetype", fig)
    end
    fig
end
