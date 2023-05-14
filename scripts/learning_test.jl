using RL
using IR
using StaticArrays
using Agents
using CairoMakie
using ColorSchemes

abm = initialise_abm(
    n_agents=4,
    norm=iNorm(195),
    utilities=(2.0,2.0,1.0,1.0)
)

abm[3].group = false
abm[4].group = false

# abm[1].policy = iNorm(255)
# abm[2].policy = iNorm(255)
# abm[1].memory = SA[true, true, true]
# abm[1].has_interacted_as_donor = true
# for _ in 1:100_000 learn!(abm[1], 2.0, abm) end

abm[1].policy









begin
for step in 1:1_000_000
    donor = abm.agents[rand(1:nagents(abm))]
    agent_step!(donor, abm)
end

begin
    fig = Figure(; resolution=(500, 500))
    ax = Axis(
        fig[1, 1];
        xlabel="Q-value Defect",
        ylabel="Q-value Cooperate",
        # xticks=ticks,
        # yticks=ticks,
        title="The distribution of stable strategies in terms of\n cooperativeness and fairness",
        titlealign=:left,
        aspect=DataAspect()
    )
    offset = 0.025
    # cmap = palette(:Hiroshige, 4)
    for (color_id, (rep, rel)) in enumerate(Iterators.product(1:2, 1:2))
        for id in 1:nagents(abm)
            scatter!(
                ax,
                abm[id].policy[rep, rel, 1],
                abm[id].policy[rep, rel, 2],
                colormap=:Hiroshige,
                colorrange = (1, 4),
                color=color_id,
                strokewidth=1
            )
        end
    end
    xlims!(ax,(0,2))
    ylims!(ax,(0,2))
    fig
end
end