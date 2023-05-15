using RL
using IR
using StaticArrays
using Agents
using CairoMakie
using ColorSchemes

norm_id = 210
abm = initialise_abm(
    n_agents=1000,
    norm=iNorm(norm_id),
    utilities=(8.0,8.0,1.0,1.0),
    proportion_incumbents_red = 0.75,
)


for step in 1:100_000_000
    donor = abm.agents[rand(1:nagents(abm))]
    agent_step!(donor, abm)
end


begin
    relation_names = ["Out-group", "In-group"]
    reputation_names = ["Bad", "Good"]
    relation_names = ["O", "I"]
    reputation_names = ["B", "G"]

    fig = Figure(; resolution=(1000, 500))
    ax_maj = Axis(
        fig[1, 1];
        xlabel="Q-value Defect",
        ylabel="Q-value Cooperate",
        title="The distribution of Q-values under Norm-$norm_id",
        titlealign=:left,
        aspect = 1,
    )
    ax_min = Axis(
        fig[1, 2];
        xlabel="Q-value Defect",
        ylabel="Q-value Cooperate",
        title="The distribution of Q-values under Norm-$norm_id",
        titlealign=:left,
        aspect = 1,
    )

    n_majority = count(agent.group == true for agent in allagents(abm))
    for (color_id, (rel, rep)) in Iterators.reverse(enumerate(Iterators.product(1:2, 1:2)))
        q_values_defect = zeros(nagents(abm))
        q_values_cooperate = zeros(nagents(abm))
        foreach(1:nagents(abm)) do id
            q_values_defect[id] = abm[id].policy[rel, rep, 1]
            q_values_cooperate[id] = abm[id].policy[rel, rep, 2]
        end
        scatter!(
            ax_maj,
            q_values_defect[1:n_majority],
            q_values_cooperate[1:n_majority],
            colormap=:Hiroshige,
            colorrange = (1, 4),
            color=color_id,
            strokewidth=1,
            label = "$(relation_names[rel]), $(reputation_names[rep])"
        )
        scatter!(
            ax_min,
            q_values_defect[n_majority+1:end],
            q_values_cooperate[n_majority+1:end],
            colormap=:Hiroshige,
            colorrange = (1, 4),
            color=color_id,
            strokewidth=1,
            label = "$(relation_names[rel]), $(reputation_names[rep])"
        )
    end
    axislegend(ax_maj, ax_maj, "Information", position = :rt)
    axislegend(ax_min, ax_min, "Information", position = :rt)
    fig
end
