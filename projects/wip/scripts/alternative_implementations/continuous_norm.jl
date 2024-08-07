using IR
using StaticArrays
using CairoMakie
using ColorSchemes

include("../norms.jl")

CairoMakie.activate!()

player_execution_mistake_rate = 0.01
judge_execution_mistake_rate = 0.01
player_perception_mistake_rate = 0.00
judge_perception_mistake_rate = 0.2
proportion_incumbents_majority = 0.5
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

function fair_norm_ess(cg, cb, dg, db; p)
    norm = reshape(SA[db, db, dg, dg, cb, cb, cg, cg], Size(2, 2, 2))
    judge = Agent(norm, p.judge_em, p.judge_pm)
    majority_strat = iStrategy(12)
    majority = Agent(majority_strat, p.maj_em, p.maj_pm)
    minority_strat = iStrategy(12)
    minority = Agent(minority_strat, p.min_em, p.min_pm)
    majority_rep, minority_rep = stationary_incumbent_reputations(
        judge, majority, minority, p.prop_maj
    )
    # @show majority_rep minority_rep
    majority_payoff, minority_payoff = incumbent_payoffs(
        majority, minority, majority_rep, minority_rep, p.prop_maj, p.utilities
    )
    prr = p_receives(majority, minority, majority_rep, p.prop_maj)
    prd = p_donates(majority, majority_rep, minority_rep, p.prop_maj)
    pbr = p_receives(minority, majority, minority_rep, 1 - p.prop_maj)
    pbd = p_donates(minority, minority_rep, majority_rep, 1 - p.prop_maj)
    fairness = prr > pbr ? pbr / prr : prr / pbr
    cooperation = p.prop_maj * prd + (1 - p.prop_maj) * pbd
    return cooperation, fairness, is_ESS(judge, majority, minority, p.prop_maj, p.utilities)
end

function fair_norm_invader(cg, cb, dg, db; p)
    norm = reshape(SA[db, db, dg, dg, cb, cb, cg, cg], Size(2, 2, 2))
    judge = Agent(norm, p.judge_em, p.judge_pm)
    majority_strat = iStrategy(12)
    majority = Agent(majority_strat, p.maj_em, p.maj_pm)
    minority_strat = iStrategy(12)
    minority = Agent(minority_strat, p.min_em, p.min_pm)
    inv = invader(judge, majority, minority, p.prop_maj, p.utilities)
    return inv === nothing ? 16 : evalpoly(2, reshape(inv[3], Size(4)))
end

let
    f((x, y)) = fair_norm_ess(1, x, 0, y; p)[3]
    reps = 0:0.0025:1
    values = map(f, Iterators.product(reps, reps))

    fig = Figure(; resolution=(600, 600))
    ax = Axis(
        fig[1, 1];
        aspect=DataAspect(),
        xlabel="Reputation of Cooperation with Bad",
        ylabel="Reputation of Defection against Bad",
        title="Which (continuous) norms are ESS with agents playing Disc?",
        subtitle="(norms have Coop-Good => 1, Defect-Good => 0)",
    )
    cmap = cgrad(:Hiroshige, 2; rev=true, categorical=true)
    hmap = heatmap!(ax, reps, reps, values; colormap=cmap)
    cb = Colorbar(fig[1, 2], hmap; nsteps=2, label="Is ESS?", ticks=0:1)

    norm_names = ["Shunning", "Image scoring", "Stern judging", "Simple standing"]
    xoffset = 0.15
    yoffset = 0.05
    positions = [
        SA[xoffset, yoffset],
        SA[1 - xoffset, yoffset],
        SA[xoffset, 1 - yoffset],
        SA[1 - xoffset, 1 - yoffset],
    ]
    for (text, position) in zip(norm_names, positions)
        text!(ax, text; position, word_wrap_with=2, align=(:center, :center))
    end
    rowsize!(fig.layout, 1, Aspect(1, 1))
    for filetype in ["png", "pdf"]
        save("figures/continuous_norms/ess.$filetype", fig)
    end
    fig
end

let
    f((x, y)) = fair_norm_ess(1, x, 0, y; p)[1]
    reps = 0:0.01:1
    values = map(f, Iterators.product(reps, reps))

    fig = Figure(; resolution=(600, 600))
    ax = Axis(
        fig[1, 1];
        aspect=DataAspect(),
        xlabel="Coop-Bad",
        ylabel="Defect-Bad",
        title="Cooperativeness of Disc agents under various norms",
        subtitle="(norms have Coop-Good => 1, Defect-Good => 0)",
    )
    cmap = cgrad(:Hiroshige; rev=true)
    hmap = heatmap!(ax, reps, reps, values; colormap=cmap) # , colormap = :plasma
    cb = Colorbar(fig[1, 2], hmap; label="Cooperativeness")
    norm_names = ["Shunning", "Image scoring", "Stern judging", "Simple standing"]
    xoffset = 0.15
    yoffset = 0.05
    positions = [
        SA[xoffset, yoffset],
        SA[1 - xoffset, yoffset],
        SA[xoffset, 1 - yoffset],
        SA[1 - xoffset, 1 - yoffset],
    ]
    for (text, position) in zip(norm_names, positions)
        text!(ax, text; position, word_wrap_with=2, align=(:center, :center))
    end
    rowsize!(fig.layout, 1, Aspect(1, 1))
    for filetype in ["png", "pdf"]
        save("figures/continuous_norms/cooperation.$filetype", fig)
    end
    fig
end

let
    f((x, y)) = fair_norm_invader(1, x, 0, y; p)
    reps = 0:0.01:1
    values = map(f, Iterators.product(reps, reps))

    fig = Figure(; resolution=(600, 600))
    ax = Axis(
        fig[1, 1];
        aspect=DataAspect(),
        xlabel="Coop-Bad",
        ylabel="Defect-Bad",
        title="Cooperativeness of Disc agents under various norms",
        subtitle="(norms have Coop-Good => 1, Defect-Good => 0)",
    )
    hmap = heatmap!(ax, reps, reps, values) # , colormap = :plasma
    display(values)
    cb = Colorbar(
        fig[1, 2],
        hmap;
        label="Invader",
        ticks=0:2:16,
        tickformat=values -> [value == 16 ? "None" : "$(Int(value))" for value in values],
    )
    norm_names = ["Shunning", "Image scoring", "Stern judging", "Simple standing"]
    xoffset = 0.15
    yoffset = 0.05
    positions = [
        SA[xoffset, yoffset],
        SA[1 - xoffset, yoffset],
        SA[xoffset, 1 - yoffset],
        SA[1 - xoffset, 1 - yoffset],
    ]
    for (text, position) in zip(norm_names, positions)
        text!(ax, text; position, word_wrap_with=2, align=(:center, :center))
    end
    rowsize!(fig.layout, 1, Aspect(1, 1))
    for filetype in ["png", "pdf"]
        save("figures/continuous_norms/invader.$filetype", fig)
    end
    fig
end
