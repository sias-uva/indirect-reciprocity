using RL
using IR
using StaticArrays
using Agents
# using CairoMakie
using GLMakie
using ColorSchemes

norm_id = 240
begin
    abm = initialise_abm(;
        n_agents=50,
        norm=iNorm(norm_id),
        utilities=(10.0, 10.0, 1.0, 1.0),
        proportion_incumbents_majority=0.80,
        exploration_rate=0.05,
        learning_rate=0.001,
    )

    n_majority = count(agent.group == false for agent in allagents(abm))

    q_values_defect = zeros(nagents(abm), 2, 2)
    q_values_cooperate = zeros(nagents(abm), 2, 2)
    for (rel, rep) in Iterators.reverse(Iterators.product(1:2, 1:2))
        for id in 1:nagents(abm)
            agent = abm[id]
            q_values_defect[id, rel, rep] = agent.policy[rel, rep, 1]
            q_values_cooperate[id, rel, rep] = agent.policy[rel, rep, 2]
        end
    end

    q_defect_majority_00 = Observable{Vector{Float64}}(q_values_defect[1:n_majority, 1, 1])
    q_defect_majority_10 = Observable{Vector{Float64}}(q_values_defect[1:n_majority, 2, 1])
    q_defect_majority_01 = Observable{Vector{Float64}}(q_values_defect[1:n_majority, 1, 2])
    q_defect_majority_11 = Observable{Vector{Float64}}(q_values_defect[1:n_majority, 2, 2])
    q_coop_majority_00 = Observable{Vector{Float64}}(q_values_cooperate[1:n_majority, 1, 1])
    q_coop_majority_10 = Observable{Vector{Float64}}(q_values_cooperate[1:n_majority, 2, 1])
    q_coop_majority_01 = Observable{Vector{Float64}}(q_values_cooperate[1:n_majority, 1, 2])
    q_coop_majority_11 = Observable{Vector{Float64}}(q_values_cooperate[1:n_majority, 2, 2])
    q_defect_minority_00 = Observable{Vector{Float64}}(
        q_values_defect[(n_majority + 1):end, 1, 1]
    )
    q_defect_minority_10 = Observable{Vector{Float64}}(
        q_values_defect[(n_majority + 1):end, 2, 1]
    )
    q_defect_minority_01 = Observable{Vector{Float64}}(
        q_values_defect[(n_majority + 1):end, 1, 2]
    )
    q_defect_minority_11 = Observable{Vector{Float64}}(
        q_values_defect[(n_majority + 1):end, 2, 2]
    )
    q_coop_minority_00 = Observable{Vector{Float64}}(
        q_values_cooperate[(n_majority + 1):end, 1, 1]
    )
    q_coop_minority_10 = Observable{Vector{Float64}}(
        q_values_cooperate[(n_majority + 1):end, 2, 1]
    )
    q_coop_minority_01 = Observable{Vector{Float64}}(
        q_values_cooperate[(n_majority + 1):end, 1, 2]
    )
    q_coop_minority_11 = Observable{Vector{Float64}}(
        q_values_cooperate[(n_majority + 1):end, 2, 2]
    )

    function scatter_q_values(ax, q_defect, q_coop, color_id, rel, rep)
        return scatter!(
            ax,
            q_defect,
            q_coop;
            colormap=(:Hiroshige, 0.5),
            colorrange=(1, 4),
            color=color_id,
            strokewidth=1,
            label="$(relation_names[rel]), $(reputation_names[rep])",
        )
    end

    begin
        # GLMakie.activate!()
        relation_names = ["Out-group", "In-group"]
        reputation_names = ["Bad", "Good"]
        relation_names = ["O", "I"]
        reputation_names = ["B", "G"]

        fig = Figure(; resolution=(1000, 500))
        ax_maj = Axis(
            fig[1, 1];
            xlabel="Q-value Defect",
            ylabel="Q-value Cooperate",
            title="Majority Q-values under Norm-$norm_id",
            titlealign=:left,
            aspect=1,
        )
        ax_min = Axis(
            fig[1, 2];
            xlabel="Q-value Defect",
            ylabel="Q-value Cooperate",
            title="Minority Q-values under Norm-$norm_id",
            titlealign=:left,
            aspect=1,
        )
        scatter_q_values(ax_maj, q_defect_majority_11, q_coop_majority_11, 4, 2, 2)
        scatter_q_values(ax_maj, q_defect_majority_01, q_coop_majority_01, 3, 1, 2)
        scatter_q_values(ax_maj, q_defect_majority_10, q_coop_majority_10, 2, 2, 1)
        scatter_q_values(ax_maj, q_defect_majority_00, q_coop_majority_00, 1, 1, 1)
        lines!(
            ax_maj, [0, 5], [0, 5]; color=:black, linestyle=:dash, label="Coop vs Defect"
        )

        scatter_q_values(ax_min, q_defect_minority_11, q_coop_minority_11, 4, 2, 2)
        scatter_q_values(ax_min, q_defect_minority_01, q_coop_minority_01, 3, 1, 2)
        scatter_q_values(ax_min, q_defect_minority_10, q_coop_minority_10, 2, 2, 1)
        scatter_q_values(ax_min, q_defect_minority_00, q_coop_minority_00, 1, 1, 1)
        lines!(
            ax_min, [0, 5], [0, 5]; color=:black, linestyle=:dash, label="Coop vs Defect"
        )

        axislegend(ax_maj, ax_maj, "Information"; position=:rt)
        axislegend(ax_min, ax_min, "Information"; position=:rt)
        # display(fig)
        limits!(ax_maj, (0, 10), (0, 10))
        limits!(ax_min, (0, 10), (0, 10))
        fig
    end

    framerate = 60
    nframes = framerate * 10
    step_iterator = 1:nframes

    record(
        fig, "figures/web/q-values_$norm_id.mp4", step_iterator; framerate=framerate
    ) do _
        for _ in 1:200
            for i in 1:nagents(abm)
                donor = abm.agents[i]
                agent_step!(donor, abm)
            end
        end
        # collect data
        q_values_defect = zeros(nagents(abm), 2, 2)
        q_values_cooperate = zeros(nagents(abm), 2, 2)
        for (rel, rep) in Iterators.reverse(Iterators.product(1:2, 1:2))
            for id in 1:nagents(abm)
                agent = abm[id]
                q_values_defect[id, rel, rep] = agent.policy[rel, rep, 1]
                q_values_cooperate[id, rel, rep] = agent.policy[rel, rep, 2]
            end
        end
        # update plot
        q_defect_majority_00[] = q_values_defect[1:n_majority, 1, 1]
        q_defect_majority_10[] = q_values_defect[1:n_majority, 2, 1]
        q_defect_majority_01[] = q_values_defect[1:n_majority, 1, 2]
        q_defect_majority_11[] = q_values_defect[1:n_majority, 2, 2]
        q_coop_majority_00[] = q_values_cooperate[1:n_majority, 1, 1]
        q_coop_majority_10[] = q_values_cooperate[1:n_majority, 2, 1]
        q_coop_majority_01[] = q_values_cooperate[1:n_majority, 1, 2]
        q_coop_majority_11[] = q_values_cooperate[1:n_majority, 2, 2]
        q_defect_minority_00[] = q_values_defect[(n_majority + 1):end, 1, 1]
        q_defect_minority_10[] = q_values_defect[(n_majority + 1):end, 2, 1]
        q_defect_minority_01[] = q_values_defect[(n_majority + 1):end, 1, 2]
        q_defect_minority_11[] = q_values_defect[(n_majority + 1):end, 2, 2]
        q_coop_minority_00[] = q_values_cooperate[(n_majority + 1):end, 1, 1]
        q_coop_minority_10[] = q_values_cooperate[(n_majority + 1):end, 2, 1]
        q_coop_minority_01[] = q_values_cooperate[(n_majority + 1):end, 1, 2]
        q_coop_minority_11[] = q_values_cooperate[(n_majority + 1):end, 2, 2]
    end
end

# begin
#     relation_names = ["Out-group", "In-group"]
#     reputation_names = ["Bad", "Good"]
#     relation_names = ["O", "I"]
#     reputation_names = ["B", "G"]

#     fig = Figure(; resolution=(1000, 500))
#     ax_maj = Axis(
#         fig[1, 1];
#         xlabel="Q-value Defect",
#         ylabel="Q-value Cooperate",
#         title="The distribution of Q-values under Norm-$norm_id",
#         titlealign=:left,
#         aspect = 1,
#     )
#     ax_min = Axis(
#         fig[1, 2];
#         xlabel="Q-value Defect",
#         ylabel="Q-value Cooperate",
#         title="The distribution of Q-values under Norm-$norm_id",
#         titlealign=:left,
#         aspect = 1,
#     )
#     for (color_id, (rel, rep)) in Iterators.reverse(enumerate(Iterators.product(1:2, 1:2)))
#         scatter!(
#             ax_maj,
#             q_defect[1:n_majority, rel, rep],
#             q_cooperate[1:n_majority, rel, rep],
#             colormap=:Hiroshige,
#             colorrange = (1, 4),
#             color=color_id,
#             strokewidth=1,
#             label = "$(relation_names[rel]), $(reputation_names[rep])"
#         )
#         scatter!(
#             ax_min,
#             q_values_defect[n_majority+1:end],
#             q_values_cooperate[n_majority+1:end],
#             colormap=:Hiroshige,
#             colorrange = (1, 4),
#             color=color_id,
#             strokewidth=1,
#             label = "$(relation_names[rel]), $(reputation_names[rep])"
#         )
#     end
#     axislegend(ax_maj, ax_maj, "Information", position = :rt)
#     axislegend(ax_min, ax_min, "Information", position = :rt)
#     fig
# end

# # Question: Modified Q-learning to 

# # maybe you can estimate the level of cooperation you will recevie given an
# # action. Frame the problem as centralised learning vs not centralised.

# # Assume

# # begin
# #     relation_names = ["Out-group", "In-group"]
# #     reputation_names = ["Bad", "Good"]
# #     relation_names = ["O", "I"]
# #     reputation_names = ["B", "G"]

# #     fig = Figure(; resolution=(1000, 500))
# #     ax_maj = Axis(
# #         fig[1, 1];
# #         xlabel="Q-value Defect",
# #         ylabel="Q-value Cooperate",
# #         title="The distribution of Q-values under Norm-$norm_id",
# #         titlealign=:left,
# #         aspect = 1,
# #     )
# #     ax_min = Axis(
# #         fig[1, 2];
# #         xlabel="Q-value Defect",
# #         ylabel="Q-value Cooperate",
# #         title="The distribution of Q-values under Norm-$norm_id",
# #         titlealign=:left,
# #         aspect = 1,
# #     )

# #     n_majority = count(agent.group == true for agent in allagents(abm))
# #     for (color_id, (rel, rep)) in Iterators.reverse(enumerate(Iterators.product(1:2, 1:2)))
# #         q_values_defect = zeros(nagents(abm))
# #         q_values_cooperate = zeros(nagents(abm))
# #         foreach(1:nagents(abm)) do id
# #             q_values_defect[id] = abm[id].policy[rel, rep, 1]
# #             q_values_cooperate[id] = abm[id].policy[rel, rep, 2]
# #         end
# #         scatter!(
# #             ax_maj,
# #             q_values_defect[1:n_majority],
# #             q_values_cooperate[1:n_majority],
# #             colormap=:Hiroshige,
# #             colorrange = (1, 4),
# #             color=color_id,
# #             strokewidth=1,
# #             label = "$(relation_names[rel]), $(reputation_names[rep])"
# #         )
# #         scatter!(
# #             ax_min,
# #             q_values_defect[n_majority+1:end],
# #             q_values_cooperate[n_majority+1:end],
# #             colormap=:Hiroshige,
# #             colorrange = (1, 4),
# #             color=color_id,
# #             strokewidth=1,
# #             label = "$(relation_names[rel]), $(reputation_names[rep])"
# #         )
# #     end
# #     axislegend(ax_maj, ax_maj, "Information", position = :rt)
# #     axislegend(ax_min, ax_min, "Information", position = :rt)
# #     fig
# # end
