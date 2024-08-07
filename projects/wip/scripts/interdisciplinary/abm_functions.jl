using Base: Fix1

"""
    simulate(norm, judge_characteristics, agent_characteristics, utilities, global_simulation_variables)

TBW
"""
function simulate(
    norm,
    judge_characteristics,
    agent_characteristics,
    utilities,
    global_simulation_variables;
    rng_seed,
    strategy_range=0:15,
)
    # Set rng
    rng = Xoshiro(rng_seed)

    # Unpack inputs
    (judge_α, judge_ε) = judge_characteristics
    (majority_α, minority_α, majority_ε, minority_ε) = agent_characteristics
    (
        population_size,
        majority_proportion,
        n_updates,
        n_interactions_per_update,
        exploration_rate,
        update_reputation_probability,
        selection_intensity,
    ) = global_simulation_variables
    (majority_benefit, majority_cost, minority_benefit, minority_cost) = utilities
    norm_matrix = iNorm(norm)

    # Preallocate some randomness that would allocate otherwise
    agents_to_interact = [
        SVector{2,Int}(rand(rng, 1:population_size, 2)) for
        _ in 1:(n_updates * n_interactions_per_update)
    ]
    agents_to_update_and_compare = [
        SVector{2,Int}(rand(rng, 1:population_size, 2)) for _ in 1:n_updates
    ]
    update_thresholds = rand(rng, n_updates)
    update_by_explore = rand(rng, n_updates) .< exploration_rate

    # Preallocate agents
    agent_strategies = rand(rng, strategy_range, population_size)
    agent_utilities = zeros(Int, population_size)
    agent_reputations = rand(rng, Bool, population_size)

    # Data collection
    df_strategy_count = DataFrame(;
        update_step=Int64[],
        strategy=Int64[],
        group=Bool[],
        prevalence=Float64[],
        group_prevalence=Float64[],
    )
    df_reputation_count = DataFrame(;
        update_step=Int64[], bad_count=Int64[], good_count=Int64[]
    )
    id_type = Tuple{Int64,Int64}
    df_interaction = DataFrame(;
        id=id_type[], donor=Int64[], recipient=Int64[], update_reputation=Bool[]
    )
    df_judgement = DataFrame(;
        id=id_type[],
        judge_info=SVector{3,Bool}[],
        judge_perceived_info=SVector{3,Bool}[],
        prob_good=Float64[],
        judgement=Bool[],
    )
    df_action = DataFrame(;
        id=id_type[],
        info=SVector{2,Bool}[],
        perceived_info=SVector{2,Bool}[],
        strategy=Int[],
        prob_coop=Float64[],
        action=Bool[],
        donor_utility_before=Int[],
        recipient_utility_before=Int[],
        donor_utility_after=Int[],
        recipient_utility_after=Int[],
    )

    # Constants
    avg_interactions_per_agent = n_interactions_per_update / population_size

    agent_groups = [n <= population_size * 0.9 for n in 1:population_size] # Custom struct if needed
    metadata = (; agent_groups)

    # Log start of simulation
    interaction_number = 0

    n_each_strategy = map(strategy_range) do s
        count(==(s), agent_strategies) / population_size
    end
    for s in strategy_range
        n_maj_tot = length(agent_strategies[agent_groups])
        n_min_tot = length(agent_strategies[.!agent_groups])
        n_maj = count(==(s), agent_strategies[agent_groups])
        n_min = count(==(s), agent_strategies[.!agent_groups])
        push!(df_strategy_count, (0, s, true, n_maj / population_size, n_maj / n_maj_tot))
        push!(df_strategy_count, (0, s, false, n_min / population_size, n_min / n_min_tot))
    end
    push!(df_reputation_count, (0, Z - count(agent_reputations), count(agent_reputations)))

    # Run simulation
    for update_step in 1:n_updates
        for interaction_step in 1:n_interactions_per_update
            # Generate the interaction id for logging
            interaction_id = (update_step, interaction_step)
            interaction_number += 1
            # Choose two random members of the population
            X, Y = agents_to_interact[interaction_number]
            # Determine the outcome of the interaction...
            is_same_group = agent_groups[X] == agent_groups[Y]
            is_good = agent_reputations[Y]
            info = SA[is_same_group, is_good]
            α = agent_groups[X] ? majority_α : minority_α
            perceived_info = mistake(α, info) .> rand(rng, SVector{2,Float64}) # preallocate?
            ε = agent_groups[X] ? majority_ε : minority_ε
            strategy = iStrategy(agent_strategies[X])
            prob_coop = (Fix1(execution_oopsie, ε) ∘ Fix1(lerp, strategy))(perceived_info)
            action = prob_coop > rand(rng)

            # After each donation game, with a probability τ, a new reputation is attributed
            # to the individual acting as donor, in accordance with the social norm fixed in
            # the population. With probability 1 − τ, the donor keeps the same reputation.

            update_reputation = rand(rng) < τ

            interaction_log = (interaction_id, X, Y, update_reputation)
            push!(df_interaction, interaction_log)
            if update_reputation
                judge_info = SA[is_same_group, is_good, action]
                judge_perceived_info =
                    mistake(judge_α, judge_info) .> rand(rng, SVector{3,Float64}) # preallocate?
                prob_good = (Fix1(mistake, judge_ε) ∘ Fix1(lerp, norm_matrix))(
                    judge_perceived_info
                )
                judgement = prob_good > rand(rng) # preallocate?
                agent_reputations[X] = judgement
                judgement_log = (
                    interaction_id, judge_info, judge_perceived_info, prob_good, judgement
                )
                push!(df_judgement, judgement_log)
            end
            cost = agent_groups[X] ? majority_cost : minority_cost
            benefit = agent_groups[Y] ? majority_benefit : minority_benefit

            donor_utility_before = agent_utilities[X]
            recipient_utility_before = agent_utilities[Y]
            agent_utilities[X] -= action * cost
            agent_utilities[Y] += action * benefit

            action_log = (
                interaction_id,
                info,
                perceived_info,
                agent_strategies[X],
                prob_coop,
                action,
                donor_utility_before,
                recipient_utility_before,
                agent_utilities[X],
                agent_utilities[Y],
            )
            push!(df_action, action_log)
        end#for
        X, Y = agents_to_update_and_compare[update_step]
        if update_by_explore[update_step]
            agent_strategies[X] = rand(rng, strategy_range)
        else
            utility_delta = (agent_utilities[Y] - agent_utilities[X])
            normalised_utility_delta = utility_delta / avg_interactions_per_agent
            update_strategy_probability = inv(
                1 + exp(-selection_intensity * normalised_utility_delta)
            )
            update_strategy = update_strategy_probability > update_thresholds[update_step]
            if update_strategy
                agent_strategies[X] = agent_strategies[Y]
            end#if
        end#if
        # Log state of simulation
        push!(
            df_reputation_count,
            (update_step, Z - count(agent_reputations), count(agent_reputations)),
        )
        for s in strategy_range
            n_maj_tot = length(agent_strategies[agent_groups])
            n_min_tot = length(agent_strategies[.!agent_groups])
            n_maj = count(==(s), agent_strategies[agent_groups])
            n_min = count(==(s), agent_strategies[.!agent_groups])
            push!(
                df_strategy_count,
                (update_step, s, true, n_maj / population_size, n_maj / n_maj_tot),
            )
            push!(
                df_strategy_count,
                (update_step, s, false, n_min / population_size, n_min / n_min_tot),
            )
        end
        # Reset utilities
        agent_utilities .= 0
    end#for
    return metadata,
    df_strategy_count, df_reputation_count, df_interaction, df_judgement,
    df_action
end

# interactions = let
#     df_intermediate = leftjoin(df_interaction, df_action; on=:id)
#     leftjoin(df_intermediate, df_judgement; on=:id)
# end

# Visualise strategy changes throughout simulation
function strategy_changes_plot(df_strategy_count)
    fig = Figure(; resolution=(600, 500))
    cmap = cgrad(:Hiroshige, 16; rev=true, categorical=true)
    ax = Axis(
        fig[1, 1];
        xlabel="Update step",
        ylabel="Prevalence",
        title="",
        # titlealign=:left,
        # aspect=DataAspect(),
    )
    # vlines!(Z:Z:n_updates;label = "Generations", color=:black)
    for s in strategy_range
        df_plot = @chain df_strategy_count begin
            subset(:strategy => ByRow(==(s)))
            groupby(:update_step)
            combine(:prevalence => sum => :prevalence)
        end
        lines!(
            ax, df_plot.update_step, df_plot.prevalence; color=cmap[s + 1], label=string(s)
        )
    end
    Colorbar(fig[1, 2]; limits=(0, 15), colormap=cmap, vertical=true, label="Strategy")
    return fig
end

function strategy_prevalence_heatmap(df_strategy_count; strategy_names)
    fig = Figure(; resolution=(600, 500))
    cmap = :viridis
    ax = Axis(
        fig[1, 1];
        xlabel="Generation",
        ylabel="Strategy",
        title="",
        yticks=1:16,
        ytickformat=x -> getindex.(Ref(strategy_names), Int.(x .- 1)),
    )
    # Group by generation
    plot_data = @chain df_strategy_count begin
        groupby([:update_step, :strategy])
        combine(:prevalence => sum => :prevalence)
        transform(
            :update_step =>
                ByRow(step -> div(step, population_size, RoundUp)) => :generation,
        )
        select(:strategy, :generation, :prevalence)
        groupby([:strategy, :generation])
        combine(:prevalence => mean => :prevalence)
        @pivot_wider(names_from = strategy, values_from = prevalence)
    end
    # display(plot_data)
    select!(plot_data, Not(:generation))
    data_matrix = Matrix{Float64}(plot_data)
    heatmap!(data_matrix; colormap=cmap, colorrange=(0, 1))
    Colorbar(fig[1, 2]; limits=(0, 1), colormap=cmap, vertical=true, label="Prevalence")

    # for filetype in ("png", "pdf")
    #     save("../figures/abm/prevalence_heatmap_$norm.$filetype", fig)
    # end
    return fig
end

function strategy_prevalence_by_group_heatmap(df_strategy_count; strategy_names)
    fig = Figure(; resolution=(600, 500))
    cmap = :viridis
    yticknames = ["Always defect", "Discriminatory", "Group-agnostic"]
    ax1 = Axis(
        fig[1, 1];
        xlabel="Generation",
        ylabel="Strategy",
        title="Majority group",
        yticks=1:3,
        ytickformat=x -> getindex.(Ref(yticknames), Int.(x)),
    )
    ax2 = Axis(
        fig[2, 1];
        xlabel="Generation",
        ylabel="Strategy",
        title="Minority group",
        yticks=1:3,
        ytickformat=x -> getindex.(Ref(yticknames), Int.(x)),
    )
    # Group by generation
    plot_data = @chain df_strategy_count begin
        transform(
            :update_step =>
                ByRow(step -> div(step, population_size, RoundUp)) => :generation,
            :strategy => ByRow(categorise_strategy) => :category,
        )
        groupby([:group, :category, :generation, :update_step])
        combine(
            [:prevalence, :group_prevalence] .=> sum .=> [:prevalence, :group_prevalence]
        )
        groupby([:group, :category, :generation])
        combine(:group_prevalence => mean => :group_prevalence)
        @pivot_wider(names_from = category, values_from = group_prevalence)
        select(Not(:generation))
        groupby(:group)
    end
    for (i, df) in enumerate(plot_data)
        data_matrix = Matrix{Float64}(select(df, Not(:group)))
        heatmap!(fig[i, 1], data_matrix; colormap=cmap, colorrange=(0, 1))
    end
    Colorbar(fig[:, 2]; limits=(0, 1), colormap=cmap, vertical=true, label="Prevalence")
    # for filetype in ("png", "pdf")
    #     save("../figures/abm/prevalence_heatmap_$norm.$filetype", fig)
    # end
    return fig
end

# # Calculate metrics:
# The average cooperation rate (ηi) in run i is computed by dividing the total
# number of cooperative acts (Ci) by the total number of donation games (Ki):

function cooperation_line_plot(df_interaction, df_action)
    fig = Figure(; resolution=(600, 500))
    cmap = cgrad(:Hiroshige, 16; rev=true, categorical=true)
    ax = Axis(
        fig[1, 1];
        xlabel="Generation",
        ylabel="Chance to cooperate",
        title="",
        # titlealign=:left,
        # aspect=DataAspect(),
    )
    plot_data = @chain df_interaction begin
        leftjoin(df_action; on=:id)
        select([:id, :donor, :recipient, :action])
        transform(
            :id => ByRow(id -> div(first(id), population_size, RoundUp)) => :generation,
            [:donor, :recipient] .=>
                ByRow(i -> agent_groups[i]) .=> [:donor_group, :recipient_group],
        )
        groupby([:generation, :donor_group, :recipient_group])
        combine(:action => mean => :cooperativeness)
    end
    line_colors = cgrad(:Hiroshige, 2; rev=true, categorical=true)
    for donor_group in (false, true)
        sdf = subset(plot_data, :donor_group => ByRow(==(donor_group)))
        for recipient_group in (false, true)
            sub_data = subset(sdf, :recipient_group => ByRow(==(recipient_group)))
            linecolor = line_colors[donor_group + 1]
            if donor_group == recipient_group
                lines!(
                    ax,
                    sub_data.generation,
                    sub_data.cooperativeness;
                    linewidth=3,
                    linestyle=:solid,
                    color=linecolor,
                )
            else
                lines!(
                    ax,
                    sub_data.generation,
                    sub_data.cooperativeness;
                    linewidth=3,
                    linestyle=:dash,
                    color=linecolor,
                )
            end
        end
    end

    color_elements = [MarkerElement(; color, marker=:rect) for color in line_colors]
    color_labels = ["Minority", "Majority"]

    style_elements = [
        LineElement(; color=:black, linestyle) for linestyle in (:solid, :dash)
    ]
    style_labels = ["In-group", "Out-group"]

    legend = Legend(
        fig[2, 1],
        [style_elements, color_elements],
        [style_labels, color_labels],
        ["Group relation", "Group"];
        nbanks=1,
        orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
    )

    for filetype in ("png", "pdf")
        save("figures/abm/cooperativeness_lineplot_$norm.$filetype", fig)
    end
    return fig
end

function average_cooperation_rate_group(df_interaction, df_action; agent_groups)
    @chain df_interaction begin
        leftjoin(df_action; on=:id)
        select([:id, :donor, :recipient, :action])
        transform(
            [:donor, :recipient] .=>
                ByRow(i -> agent_groups[i]) .=> [:donor_group, :recipient_group],
        )
        groupby([:donor_group, :recipient_group])
        combine(:action => mean => :mean_cooperation_rate)
        transform(
            [:donor_group, :recipient_group] .=>
                ByRow(x -> x ? "Majority" : "Minority") .=>
                    [:donor_group, :recipient_group],
        )
        @pivot_wider(names_from = recipient_group, values_from = mean_cooperation_rate)
    end
end

average_cooperation_rate(df_action) = mean(df_action.action)
# average_cooperation_rate = mean(df_action.action)

# Average fraction of Good and Bad reputations
function average_reputation(df_reputation_count; population_size)
    return mean(df_reputation_count.good_count) / population_size
end
# average_reputation = mean(df_reputation_count.good_count)/population_size

# Average time spent playing each strategy
function average_time_each_strategy(df_strategy_count)
    return @chain df_strategy_count begin
        groupby([:strategy, :group])
        combine(:prevalence => mean => :prevalence)
        @pivot_wider(names_from = group, values_from = prevalence)
        rename("false" => :minority, "true" => :majority)
    end
end

# function pair_prevalence(df_strategy_count)
#     df1 = select(df_strategy_count, Not(:prevalence))
#     df1_maj = @chain df1 begin
#         subset(:group)
#         select(Not(:group))
#     end
#     df1_min = @chain df1 begin
#         subset(:group => ByRow(!))
#         select(Not(:group))
#     end
#     pair_df = mapreduce(vcat, zip(groupby(df1_maj, :update_step), groupby(df1_min, :update_step))) do (df_maj, df_min)
#         df_renamed_maj = rename(df_maj, :strategy => :strategy_maj, :group_prevalence => :maj_prevalence)
#         df_renamed_min = rename(df_min, :strategy => :strategy_min, :group_prevalence => :min_prevalence)
#         df_renamed_min = select(df_renamed_min, Not(:update_step))
#         df_cross = crossjoin(df_renamed_maj, df_renamed_min)
#         select(df_cross, :update_step, :strategy_maj, :strategy_min, [:maj_prevalence, :min_prevalence] => ByRow((x, y) -> x * y) => :joint_prevalence)
#     end
#     return combine(groupby(pair_df, [:strategy_maj, :strategy_min]), :joint_prevalence => mean => :avg_joint_prevalence)
# end

# pair_prevalence(df_strategy_count)

"""
    average_fairness(df_interaction, df_action)

Ratio between best and worst group's expected/average payoffs over simulation
duration.
"""
function average_fairness(df_interaction, df_action; agent_groups)
    # Join interaction and action on id
    joined_df = leftjoin(df_interaction, df_action; on=:id)
    select!(
        joined_df,
        :donor,
        :recipient,
        :donor_utility_before,
        :donor_utility_after,
        :recipient_utility_before,
        :recipient_utility_after,
    )

    transform!(
        joined_df,
        [:donor, :recipient] .=>
            ByRow(Fix1(getindex, agent_groups)) .=> [:donor_group, :recipient_group],
    ) #  

    transform!(
        joined_df,
        [:donor_utility_before, :donor_utility_after] =>
            ((x, y) -> y .- x) => :donor_utility_change,
    )
    transform!(
        joined_df,
        [:recipient_utility_before, :recipient_utility_after] =>
            ((x, y) -> y .- x) => :recipient_utility_change,
    )
    grouped_df = groupby(joined_df, [:donor_group, :recipient_group])
    combined_df = combine(
        grouped_df,
        [:donor_utility_change, :recipient_utility_change] .=>
            sum .=> [:total_donor_utility, :total_recipient_utility],
    )
    donor_df = select(
        combined_df, [:donor_group => :group, :total_donor_utility => :total_utility]
    )
    recipient_df = select(
        combined_df,
        [:recipient_group => :group, :total_recipient_utility => :total_utility],
    )
    total_df = combine(
        groupby(vcat(donor_df, recipient_df), :group),
        :total_utility => sum => :total_utility,
    )
    sort!(total_df, :group)

    min_per_capita = total_df[1, :total_utility] / (population_size - count(agent_groups))
    maj_per_capita = total_df[2, :total_utility] / count(agent_groups)
    lower, higher = extrema((min_per_capita, maj_per_capita))
    return lower / higher
end
