using IR
using StaticArrays
using DataFrames
using CairoMakie
using ColorSchemes

begin
player_execution_mistake_rate = 0.21
judge_execution_mistake_rate = 0.01
player_perception_mistake_rate = 0.0
judge_perception_mistake_rate = 0.0
proportion_incumbents_red = 0.9
utilities = SA[2, 2, 1, 1]

p = (;
    rem=player_execution_mistake_rate,
    bem=player_execution_mistake_rate,
    jem=judge_execution_mistake_rate,
    rpm=player_perception_mistake_rate,
    bpm=player_perception_mistake_rate,
    jpm=judge_perception_mistake_rate,
    pR=proportion_incumbents_red,
    utilities=utilities
)

function get_agents(norm_i, rs_i, bs_i; p)
    norm = iNorm(norm_i)
    judge = Agent(norm, p.jem, p.jpm)
    red_strat = iStrategy(rs_i)
    red = Agent(red_strat, p.rem, p.rpm)
    blue_strat = iStrategy(bs_i)
    blue = Agent(blue_strat, p.bem, p.bpm)
    return judge, red, blue
end

function find_ESS(; p)
    norm_ints = 0:255
    strategy_ints = 0:15
    df = rename!(
        DataFrame(Iterators.product(norm_ints, strategy_ints, strategy_ints)),
        [:norm, :red_strat, :blue_strat],
    )
    df[!, :is_ess] = map(eachrow(df)) do row
        judge, red, blue = get_agents(row...; p)
        is_ESS(judge, red, blue, p.pR, p.utilities)
    end
    return df
end

df = find_ESS(; p)

# Uncomment cbar stuff for colors based on sum of number of cooperative bits
function generate_quadrant_df(_df; p)
    df = deepcopy(_df)
    info_df = DataFrame([:red_rep, :blue_rep, :red_payoff, :blue_payoff, :fairness, :cooperation, :prr, :pbr, :prd, :pbd] .=> Ref(Float64[]))
    quadrants = Int8[]
    foreach(eachrow(df)) do row
        n, r, b, _ = row
        judge, red, blue = get_agents(n, r, b; p)
        red_rep, blue_rep = stationary_incumbent_reputations(judge, red, blue, p.pR)
        red_payoff, blue_payoff = incumbent_payoffs(red, blue, red_rep, blue_rep, p.pR, p.utilities)
        prr = p_receives(red, blue, red_rep, p.pR)
        prd = p_donates(red, red_rep, blue_rep, p.pR)
        pbr = p_receives(blue, red, blue_rep, 1 - p.pR)
        pbd = p_donates(blue, blue_rep, red_rep, 1 - p.pR)
        fairness = prr > pbr ? pbr / prr : prr / pbr
        cooperation = p.pR * prd + (1 - p.pR) * pbd
        push!(info_df, (red_rep, blue_rep, red_payoff, blue_payoff, fairness, cooperation, prr, pbr, prd, pbd))
        push!(quadrants, (cooperation < 0.5) + 2(fairness < 0.5))
    end
    df = hcat(df, info_df, quadrants)
    rename!(df, :x1 => :quadrant)
    subset!(df, :is_ess)
    subset!(df, :red_strat => ByRow(!=(0)))
    # unique!(df, [:red_payoff, :blue_payoff, :red_rep, :blue_rep])
    return df
end

let
    quadrant_df = generate_quadrant_df(df; p)
    quadrant_size = map(0:3) do quadrant_i
        sdf = subset(quadrant_df, :quadrant => ByRow(==(quadrant_i)))
        size(sdf, 1)
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
        aspect=DataAspect()
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
        color=quadrant_df.blue_payoff,
        strokewidth=1
    )
    hlines!(ax, 0.5; color=:black, linestyle=:dash)
    vlines!(ax, 0.5; color=:black, linestyle=:dash)
    Colorbar(fig[1, 2], sc; label="Minority payoff")
    quadrant_text = (
        "Everyone wins", "Most lose,\nsome win", "Most win,\nsome lose", "Everyone loses"
    )
    quadrant_labels = [text * "\n(n=$n)" for (text, n) in zip(quadrant_text, quadrant_size)]
    x_offset = 0.25
    x_offsets = (x_offset, -x_offset, x_offset, -x_offset)
    y_offset = 0.1
    y_offsets = (y_offset, y_offset, -y_offset, -y_offset)
    for (quadrant_label, x_offset, y_offset) in zip(quadrant_labels, x_offsets, y_offsets)
        text!(
            ax,
            quadrant_label;
            position=(0.5 + x_offset, 0.5 + y_offset),
            word_wrap_with=2,
            align=(:center, :center)
        )
    end
    rowsize!(fig.layout, 1, Aspect(1, 1))
    #save("./scripts/figures/pnas/coop_fairness_payoff_scatter.pdf", fig)
    fig
end
end

let
    scatter_df = generate_quadrant_df(df; p)
    scatter_df.quadrant = map(eachrow(scatter_df)) do row
        (row.pbd > 0.5) + 2(row.prd > 0.5)
    end
    quadrant_size = map(0:3) do quadrant_i
        sdf = subset(scatter_df, :quadrant => ByRow(==(quadrant_i)))
        size(sdf, 1)
    end
    ticks = 0:0.25:1
    fig = Figure(; resolution=(500, 500))
    ax = Axis(
        fig[1, 1];
        xlabel="Cooperativeness of majority group",
        ylabel="Cooperativeness of minority group",
        xticks=ticks,
        yticks=ticks,
        title="Categorising the cooperativeness of\nstable strategies",
        titlealign=:left,
        aspect=DataAspect()
    )
    offset = 0.025
    lims = (0 - offset, 1 + offset)
    limits!(ax, lims, lims)
    cmap = cgrad(:Hiroshige; rev=true)
    sc = scatter!(
        ax,
        scatter_df.prd,
        scatter_df.pbd;
        colormap=cmap,
        color=scatter_df.blue_payoff,
        strokewidth=1
    )
    ## Line f(x) = x to show equal cooperation
    lines!(ax, 0:0.1:1, x -> x, color=:black, linestyle=:dash, label="Fair cooperation")
    axislegend(ax; position = :lt)
    # text!(
    #     ax,
    #     "Minority more\ncooperative";
    #     position=(0.25, 0.75),
    #     word_wrap_with=2,
    #     align=(:center, :center)
    # )
    # text!(
    #     ax,
    #     "Majority more\ncooperative";
    #     position=(0.75, 0.25),
    #     word_wrap_with=2,
    #     align=(:center, :center)
    # )
    
    ## Colorbar
    Colorbar(fig[1, 2], sc; label="Minority payoff")
    ## Quadrants
    # begin
    #     hlines!(ax, 0.5; color=:black, linestyle=:dash)
    #     vlines!(ax, 0.5; color=:black, linestyle=:dash)
    #     quadrant_text = (
    #         "Universal\ncooperation", "", "Majority cooperation", "No one\ncooperates"
    #     )
    #     quadrant_labels = [text * "\n(n=$n)" for (text, n) in zip(quadrant_text, quadrant_size)]
    #     x_offset = 0.25
    #     x_offsets = (x_offset, -x_offset, x_offset, -x_offset)
    #     y_offset = 0.1
    #     y_offsets = (y_offset, y_offset, -y_offset, -y_offset)
    #     for (quadrant_label, x_offset, y_offset, n) in zip(quadrant_labels, x_offsets, y_offsets, quadrant_size)
    #         continue
    #         n == 0 && continue
    #         text!(
    #             ax,
    #             quadrant_label;
    #             position=(0.5 + x_offset, 0.5 + y_offset),
    #             word_wrap_with=2,
    #             align=(:center, :center)
    #         )
    #     end
    # end
    rowsize!(fig.layout, 1, Aspect(1, 1))
    for filetype in ("pdf", "png")
        #save("./scripts/figures/pnas/coop_each_group_scatter.$filetype", fig)
    end
    fig
end

let
    insularity_df = generate_quadrant_df(df; p)
    info_df = DataFrame([:red_insularity, :blue_insularity] .=> Ref(Float64[]))
    foreach(eachrow(insularity_df)) do row
        norm, red_strat, blue_strat, _, red_rep, blue_rep = row
        judge, red, blue = get_agents(norm, red_strat, blue_strat; p)
        pRdR = lerp(SA[red(1, 0), red(1, 1)], red_rep)
        pRdB = lerp(SA[red(0, 0), red(0, 1)], blue_rep)
        pBdR = lerp(SA[blue(0, 0), blue(0, 1)], red_rep)
        pBdB = lerp(SA[blue(1, 0), blue(1, 1)], blue_rep)
        red_insularity = pRdR / (pRdR + pRdB)
        blue_insularity = pBdB / (pBdB + pBdR)
        push!(info_df, (red_insularity, blue_insularity))
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
        aspect=DataAspect()
    )
    offset = 0.025
    lims = (0 - offset, 1 + offset)
    limits!(ax, lims, lims)
    cmap = cgrad(:Hiroshige; rev=true)
    sc = scatter!(
        ax,
        insularity_df.red_insularity,
        insularity_df.blue_insularity;
        colormap=cmap,
        color=insularity_df.blue_payoff,
        strokewidth=1
    )
    
    ## Colorbar
    Colorbar(fig[1, 2], sc; label="Minority payoff")

    rowsize!(fig.layout, 1, Aspect(1, 1))
    for filetype in ("pdf", "png")
        #save("./scripts/figures/pnas/insularity_scatter.$filetype", fig)
    end
    fig
end

let
    subset(
        df,
        :is_ess, 
        # :red_strat  => ByRow(!=(0))
        [:red_strat, :blue_strat]  => ByRow((x,y) -> x!=y)
    )
    quadrant_df = generate_quadrant_df(df; p)
    subset(quadrant_df, :quadrant => ByRow(==(0)))
end