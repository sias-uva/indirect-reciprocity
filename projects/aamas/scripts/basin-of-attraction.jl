using Statistics
using StatsBase
using Random
using Base: Fix1, Fix2
using CSV

using StaticArrays
using DataFrames
using CairoMakie
using Tidier

using IR
using IRUtils

include("rl-functions.jl")

const Policy = SArray{Tuple{2,2,2},Float64,3,8}

function train_and_collect!(abm::IRABM{AgentDataRL}, data)
    agent_data = abm.agent_data
    properties = abm.properties
    rng = abm.rng
    for interaction_number in 1:(properties.n_training_interactions)
        X, Y = properties.agents_to_interact_training[interaction_number]
        # Determine the action taken by the agent
        is_same_group = agent_data.groups[X] == agent_data.groups[Y]
        is_good = agent_data.reputations[Y]
        info = SA[is_same_group, is_good]
        action, perceived_info = act(X, info, abm)
        agent_data.interacted_as_donor[X] = true # Update that the chosen donor has now been a donor
        push!(data.df_cooperation, (X, Y, action))
        # After each donation game, with a probability τ, a new reputation is attributed
        # to the individual acting as donor, in accordance with the social norm fixed in
        # the population. With probability 1 − τ, the donor keeps the same reputation.
        update_reputation = rand(rng) < properties.update_reputation_probability # τ
        if update_reputation
            judge_info = SA[is_same_group, is_good, action]
            judgement = judge(judge_info, abm)
            agent_data.reputations[X] = judgement
            agent_data.memories[X] = SA[perceived_info..., action]
        end
        cost = agent_data.groups[X] ? properties.majority_cost : properties.minority_cost
        benefit =
            agent_data.groups[Y] ? properties.majority_benefit : properties.minority_benefit
        for (A, utility) in zip((X, Y), (action * -cost, action * benefit))
            new_policy = learn(A, utility, abm)
            agent_data.policies[A] = new_policy
            agent_data.utilities[A] += utility
        end
        if mod(interaction_number, properties.generation_length) == 0
            push!(data.policies, copy(agent_data.policies))
        end
    end
end

function initialise_rlabm_with_policies(
    norm,
    judge_characteristics,
    agent_characteristics,
    utilities,
    global_simulation_variables;
    rng,
    agent_policies,
)
    # Calculate some auxiliary properties based on inputs
    population_size,
    majority_proportion, n_training_interactions, n_data_interactions,
    _... = global_simulation_variables
    norm_matrix = iNorm(norm)
    # (preallocate some randomness that would allocate otherwise)
    agents_to_interact_training = [
        SVector{2,Int}(rand(rng, 1:population_size, 2)) for _ in 1:n_training_interactions
    ]
    agents_to_interact_data = [
        SVector{2,Int}(rand(rng, 1:population_size, 2)) for _ in 1:n_data_interactions
    ]
    properties = merge(
        (; norm, norm_matrix),
        judge_characteristics,
        agent_characteristics,
        global_simulation_variables,
        utilities,
        (; agents_to_interact_training, agents_to_interact_data),
    )

    majority_population_size = floor(Int, majority_proportion * population_size)
    agent_groups = AgentGroups(majority_population_size, population_size)
    agent_utilities = zeros(Int, population_size)
    agent_reputations = rand(rng, Bool, population_size)
    agent_memories = zeros(SVector{3,Bool}, population_size)
    interacted_as_donor = falses(population_size)
    n_donor_interactions = zeros(population_size)
    n_cooperations = zeros(population_size)

    agent_data = AgentDataRL(
        agent_groups,
        agent_policies,
        agent_utilities,
        agent_reputations,
        agent_memories,
        interacted_as_donor,
        n_donor_interactions,
        n_cooperations,
    )
    return IRABM(agent_data, properties, rng)
end

# for norm in (0, 150, 192, 195, 243)
begin # Set ABM parameters
    norm = 195
    # Magic constants
    strategy_range = 0:15

    # Global simulation variables
    Z = population_size = 50
    majority_proportion = 0.9
    n_generations = Int(population_size * 10)
    generation_length = 10 * population_size
    n_training_interactions = n_generations * generation_length
    n_data_interactions = n_training_interactions
    μ = exploration_rate = 1 / 10 # 1 / population_size
    τ = update_reputation_probability = 0.9
    β = selection_intensity = 1
    learning_rate = 0.1

    global_simulation_variables = (;
        population_size,
        majority_proportion,
        n_training_interactions,
        n_data_interactions,
        exploration_rate,
        update_reputation_probability,
        learning_rate,
        generation_length,
    )

    # Characteristics of players (rate of errors)
    majority_α = SA[0.00, 0.00]
    minority_α = SA[0.00, 0.00]
    majority_ε = 0.01
    minority_ε = 0.01
    agent_characteristics = (; majority_α, minority_α, majority_ε, minority_ε)

    # Characteristics of judge and norm used
    judge_α = SA[0.00, 0.00, 0.00]
    judge_ε = 0.01
    judge_characteristics = (; judge_α, judge_ε)

    # The costs and benefits of interacting
    majority_benefit = 10
    minority_benefit = 10
    majority_cost = 1
    minority_cost = 1
    utilities = (; majority_benefit, majority_cost, minority_benefit, minority_cost)
    p = (;
        maj_em=majority_ε,
        min_em=minority_ε,
        judge_em=judge_ε,
        maj_pm=majority_α,
        min_pm=minority_α,
        judge_pm=judge_α,
        prop_maj=majority_proportion,
        utilities=SA[majority_benefit, minority_benefit, majority_cost, minority_cost],
    )
end

begin # Main
    starting_policy_scale = 3
    population_size = global_simulation_variables.population_size
    granular_output = DataFrame(
        :n_seeded_agents => Int[],
        :seed => Int[],
        :cooperation => Float64[],
        :fairness => Float64[],
        :margin_distance => Vector{Float64}[],
        # :final_policies => Vector{Policy}[],
        :policies_per_generation => Vector{Vector{Policy}}[],
    )
    n_seeded_agents_range = 0:(population_size ÷ 2):population_size
    n_seeded_agents_range = [0]
    n_runs = 50
    for n_seeded_agents in n_seeded_agents_range
        seeds = 1:n_runs
        coop_fairness_vec = zeros(SVector{2,Float64}, n_runs)
        @show n_seeded_agents
        for (i_seed, seed) in enumerate(seeds)
            # @show i_seed
            rng = Xoshiro(seed)
            # To begin with all agents have a random policy policy
            ideal_policy_maj, ideal_policy_min =
                starting_policy_scale .* get_ideal_policies(norm; p, rng)
            agent_policies = starting_policy_scale * rand(rng, Policy, population_size)
            # Then we replace some with AllD
            if n_seeded_agents != 0
                seeded_agent_policies = [
                    starting_policy_scale * i_to_pol(0) for i in 1:n_seeded_agents
                ]
                seeded_agents = sample(
                    rng, 1:population_size, n_seeded_agents; replace=false
                )
                for (agent, seeded_policy) in zip(seeded_agents, seeded_agent_policies)
                    agent_policies[agent] = seeded_policy
                end
            end
            abm = initialise_rlabm_with_policies(
                norm,
                judge_characteristics,
                agent_characteristics,
                utilities,
                global_simulation_variables;
                rng,
                agent_policies,
            )
            data = (;
                df_cooperation=DataFrame(
                    :donor => Int64[], :recipient => Int64[], :action => Bool[]
                ),
                policies=Vector{Policy}[],
            )
            train_and_collect!(abm, data)
            coop_fairness = get_coop_fairness(abm, data)
            coop_fairness_vec[i_seed] = coop_fairness
            sum_of_margins = get_sum_of_margins_over_time(
                abm, data; ideal_strategies=pol_to_i.((ideal_policy_maj, ideal_policy_min))
            )
            # final_policies = copy(abm.agent_data.policies)
            policies_this_generation = data.policies
            push!(
                granular_output,
                (
                    n_seeded_agents,
                    seed,
                    coop_fairness...,
                    sum_of_margins,
                    policies_this_generation,
                ),
            )
        end
    end
    granular_output
end

# Plot the path of each group's average Q-values for each state/information.
begin
    df_policies = DataFrame(
        :n_seeded_agents => Int64[],
        :seed => Int64[],
        :generation => Int64[],
        :agent => Int64[],
        :policy => Policy[],
    )
    policies_generation_runs = granular_output.policies_per_generation
    for (run, policies_generations) in enumerate(policies_generation_runs)
        for (generation, policies) in enumerate(policies_generations)
            for (agent, policy) in enumerate(policies)
                push!(
                    df_policies,
                    (
                        granular_output.n_seeded_agents[run],
                        granular_output.seed[run],
                        generation,
                        agent,
                        policy,
                    ),
                )
            end
        end
    end
    # transform!(
    #     df_policies,
    #     :agent => ByRow(x -> x <= majority_proportion * population_size) => :agent_group,
    # )
end

begin
    df_policies.agent_group .= df_policies.agent .<= majority_proportion * population_size
    optimal_combinations = get_optimal_strategy_combinations(norm; p)
    df_policies.similarity .=
        similarity_to_optimal.(
            df_policies.policy, df_policies.agent_group; optimal_combinations
        )
    df_policies
end

let
    fig = Figure(; resolution=(600, 600))
    ax = Axis(
        fig[1, 1];
        xlabel="Generation",
        ylabel="Distance from Q-value to margin",
        title="Minimum L1 distance between Q-values and line x=y\nfor multiple runs: Norm-$norm",
    )
    cdict = Dict(n_seeded_agents_range .=> 1:length(n_seeded_agents_range))
    cmap = cgrad(:Egypt, length(n_seeded_agents_range); categorical=true)
    smap = Dict(false => :dash, true => :solid)
    alpha = 0.1
    df_margin = @chain df_policies begin
        groupby([:n_seeded_agents, :generation, :seed])
        combine(:similarity => minimum)
    end
    display(df_margin)
    foreach(groupby(df_margin, [:n_seeded_agents, :seed])) do sdf
        transform!(sdf, :n_seeded_agents => ByRow(x -> (cmap[cdict[x]], alpha)) => :rgb)
        lines!(ax, sdf.generation, sdf.similarity_minimum; color=sdf.rgb, label=nothing)
    end
    df_margin_group = combine(
        groupby(df_margin, [:n_seeded_agents, :generation]),
        :similarity_minimum => minimum => :similarity_minimum_group,
    )
    transform!(
        df_margin_group,
        :n_seeded_agents => ByRow(x -> cmap[cdict[x]]) => :rgb,
        # :agent_group => ByRow(x -> smap[x]) => :linestyle
    )
    foreach(groupby(df_margin_group, :n_seeded_agents)) do sdf
        lines!(ax, sdf.generation, sdf.similarity_minimum_group; color=sdf.rgb)
    end
    begin
        color_lines = [
            LineElement(; color=cmap[i]) for (i, _) in enumerate(n_seeded_agents_range)
        ]
        transparency_markers = [
            MarkerElement(;
                color=(:black, alpha), marker=:rect, markersize=15, strokewidth=1
            ) for alpha in (1, 0.1)
        ]
        objects = [color_lines, transparency_markers]
        object_labels = [string.(n_seeded_agents_range), ["Over all runs", "This run"]]
        legend_titles = ["Number of\nAllD agents", "Which minimum?"]
        Legend(fig[1, 2], objects, object_labels, legend_titles)
    end
    for filetype in ("png", "pdf")
        # save(
        #     "projects/aamas/figures/basin-of-attraction/margin-distance-$(norm).$filetype",
        #     fig,
        # )
    end
    display(fig)
    nothing
end

# let
#     fig = Figure(; resolution=(600, 600))
#     ax_maj = Axis(
#         fig[1, 1]; xlabel="Q-value defect", ylabel="Q-value cooperate", title="Majority"
#     )
#     ax_min = Axis(
#         fig[1, 2]; xlabel="Q-value defect", ylabel="Q-value cooperate", title="Minority"
#     )
#     linkaxes!(ax_maj, ax_min)
#     for ax in (ax_maj, ax_min)
#         ablines!(ax, 0, 1; color=:black, linestyle=:dash)
#     end
#     fig[0, :] = Label(
#         fig, "Final Q-values in one run of Norm-$norm"; word_wrap=true, font=:bold
#     )
#     # first index is the run (can be any), last index is the generation.
#     policies = granular_output.policies_per_generation[1][end]
#     # policies = granular_output.final_policies[1]
#     cmap = cgrad(:Egypt, 4; categorical=true)
#     for (agent, policy) in enumerate(policies)
#         for I in CartesianIndices((1:2, 1:2))
#             linear_I = evalpoly(2, Tuple(I) .- 1) + 1
#             x, y = policy[I, :]
#             ax = agent <= population_size * majority_proportion ? ax_maj : ax_min
#             scatter!(ax, Point2(x, y); color=cmap[linear_I], strokewidth=1.0)
#         end
#     end
#     color_elements = [
#         MarkerElement(; color, marker=:rect, markersize=10, strokewidth=1) for
#         color in getindex.(Ref(cmap), 1:4)
#     ]
#     color_labels = ["Out-bad", "In-bad", "Out-good", "In-good"]
#     legend = Legend(
#         fig[2, :],
#         [color_elements],#
#         [color_labels],#
#         ["Information"];
#         nbanks=2,
#         orientation=:horizontal,
#         tellwidth=false,
#         tellheight=true,
#     )
#     for filetype in ("png", "pdf")
#         save(
#             "projects/aamas/figures/basin-of-attraction/final-q-value-scatter-$(norm).$filetype", fig
#         )
#     end
#     display(fig)
# end
# end

let
    fig = Figure(; size=(450, 500))
    ax_maj = Axis(
        fig[1, :];
        xlabel="Q-value defect",
        ylabel="Q-value cooperate",
        title="Majority",
        # aspect=1,
    )
    ax_min = Axis(
        fig[2, :];
        xlabel="Q-value defect",
        ylabel="Q-value cooperate",
        title="Minority",
        # aspect=1,
        # limits = ((0, 1), (0, 1)),
    )
    linkaxes!(ax_maj, ax_min)
    some = 1
    cmap = cgrad(:Egypt, 4; categorical=true)
    info_color_dict = Dict((:ob, :ib, :og, :ig) .=> cmap)
    df_plot = @chain df_policies begin
        groupby([:n_seeded_agents, :generation, :agent_group])
        combine(:policy => mean)
        transform(:policy_mean => ByRow(pol_to_points) => AsTable)
        select(Not(:policy_mean))
    end
    subset!(df_plot, :n_seeded_agents => ByRow(==(n_seeded_agents_range[some])))
    for (i, ax) in enumerate((ax_maj, ax_min))
        group = 2 - i
        sdf = subset(df_plot, :agent_group => ByRow(==(group)))
        for col in (:ob, :ib, :og, :ig)
            color = info_color_dict[col]
            lines!(ax, sdf[!, col]; color)
            scatter!(
                ax,
                sdf[end, col];
                color,
                strokecolor=:black,
                strokewidth=1.0,
                markersize=7.5,
            )
        end
        ablines!(ax, 0, 1; color=:black, linestyle=:dash)
    end
    # limits!(ax, (0, starting_policy_scale), (0, starting_policy_scale))
    color_elements = [
        MarkerElement(; color, marker=:rect, markersize=10, strokewidth=1) for
        color in getindex.(Ref(cmap), 1:4)
    ]
    color_labels = ["Out-bad", "In-bad", "Out-good", "In-good"]
    legend = Legend(
        fig[1:end, end + 1],
        [color_elements],#
        [color_labels],#
        ["Information"];
        # nbanks=2,
        orientation=:vertical,
        tellwidth=false,
        tellheight=true,
    )
    fig[0, :] = Label(
        fig,
        # "Average path taken by Q-values with $(n_seeded_agents_range[some]) of $population_size agents seeded AllD under norm-$norm";
        rich("Average path of Q-values under ", rich("SternJudging"; font=:bold_italic));
        fontsize=18,
        # word_wrap=true,
        justification=:left,
        halign=:left,
        font=:bold,
        # tellwidth=false,
        width=Relative(2),
    )
    rowsize!(fig.layout, 0, Relative(0.05))
    colsize!(fig.layout, 2, Relative(0.3))
    resize_to_layout!(fig)
    for filetype in ("png", "pdf")
        save(
            "projects/aamas/figures/basin-of-attraction/q-value-path-$(norm).$filetype", fig
        )
    end
    display(fig)
end

let
    fig = Figure(; resolution=(600, 500))
    ax_maj = Axis(
        fig[1, 1];
        xlabel="Q-value defect",
        ylabel="Q-value cooperate",
        title="Majority",
        aspect=1,
    )
    ax_min = Axis(
        fig[1, 2];
        xlabel="Q-value defect",
        ylabel="Q-value cooperate",
        title="Minority",
        aspect=1,
    )
    linkaxes!(ax_maj, ax_min)
    some = 3
    fig[0, :] = Label(
        fig,
        "Average path taken by Q-values with $(n_seeded_agents_range[some]) of $population_size agents seeded AllD under norm-$norm";
        word_wrap=true,
        font=:bold,
    )
    cmap = cgrad(:Egypt, 4; categorical=true)
    info_color_dict = Dict((:ob, :ib, :og, :ig) .=> cmap)
    df_plot = @chain df_policies begin
        subset(:generation => ByRow(==(n_generations)))
        sort!(:similarity)
        groupby([:n_seeded_agents, :agent_group])
        combine(sdf -> sdf[1, :])
        select(Not(:similarity, :generation, :policy))
        leftjoin(df_policies; on=[:n_seeded_agents, :seed, :agent_group, :agent])
        groupby([:n_seeded_agents, :agent_group, :generation])
        combine(:policy => mean => :policy)
        transform(:policy => ByRow(pol_to_points) => AsTable)
    end
    subset!(df_plot, :n_seeded_agents => ByRow(==(n_seeded_agents_range[some])))
    for (i, ax) in enumerate((ax_maj, ax_min))
        group = i - 1
        sdf = subset(df_plot, :agent_group => ByRow(==(group)))
        for col in (:ob, :ib, :og, :ig)
            color = info_color_dict[col]
            lines!(ax, sdf[!, col]; color)
            scatter!(
                ax,
                sdf[end, col];
                color,
                strokecolor=:black,
                strokewidth=1.0,
                markersize=7.5,
            )
        end
        ablines!(ax, 0, 1; color=:black, linestyle=:dash)
    end
    limits!((0, starting_policy_scale), (0, starting_policy_scale))
    color_elements = [
        MarkerElement(; color, marker=:rect, markersize=10, strokewidth=1) for
        color in getindex.(Ref(cmap), 1:4)
    ]
    color_labels = ["Out-bad", "In-bad", "Out-good", "In-good"]
    legend = Legend(
        fig[2, :],
        [color_elements],#
        [color_labels],#
        ["Information"];
        nbanks=1,
        orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
    )
    resize_to_layout!(fig)
    for filetype in ("png", "pdf")
        save(
            "projects/aamas/figures/basin-of-attraction/worst-q-value-path-$(norm).$filetype",
            fig,
        )
    end
    display(fig)
end
# end
# end
# look at the 3 worst agents at the end and retrace their steps/Q-values.
# maybe transparent?

# look at situations where defection is learned, change the seed once the
# q-values are initiated and see whether defection is still learned

# if not recovering then it is not random, if it recoves then it is random.

# What happens to the Q-values in the green cases in Fig 1? Why are there two
# "regimes"?

let
    df_margin = @chain df_policies begin
        groupby([:n_seeded_agents, :generation, :seed])
        combine(:similarity => minimum)
    end
    @chain df_policies begin
        subset(:generation => ByRow(==(n_generations)))
        select(Not([:generation, :policy, :similarity]))
        leftjoin(df_policies; on=[:n_seeded_agents, :seed, :agent_group, :agent])
    end
end

function similarity_to_optimal_expand(policy, group; optimal_combinations)
    group_index = 2 - group
    _, optimal_index = findmax(optimal_combinations) do optimal_combination
        strategy = optimal_combination[group_index]
        istrategy = iStrategy(strategy)
        similarity_score(policy, istrategy)
    end
    optimal_strategy = iStrategy(optimal_combinations[optimal_index][group_index])
    return similarity_score_expand(policy, optimal_strategy)
end

function similarity_score_expand(policy, strategy)
    scores = map(CartesianIndices((1:2, 1:2))) do I
        @views q_pair = policy[I, :]
        @views optimal = strategy[I]
        similarity_score_q(q_pair, optimal)
    end
    keys = (:ob, :ib, :og, :ig)
    values = (scores[I] for I in CartesianIndices((1:2, 1:2)))
    return (; zip(keys, values)...)
end

# TODO: do the things below because this function works.

begin
    df_expand = @chain df_policies begin
        transform(
            [:policy, :agent_group] =>
                ByRow((x, y) -> similarity_to_optimal_expand(x, y; optimal_combinations)) =>
                    AsTable,
        )
    end
end

let
    fig = Figure(; resolution=(600, 600))
    ax = Axis(
        fig[1, 1];
        xlabel="Generation",
        ylabel="Distance from Q-value to margin",
        title="Minimum L1 distance between Q-values and line x=y\nfor multiple runs: Norm-$norm",
    )
    cdict = Dict(n_seeded_agents_range .=> 1:length(n_seeded_agents_range))
    cmap = cgrad(:Egypt, 4; categorical=true)
    # smap = Dict(false => :dash, true => :solid)
    alpha = 0.1
    df_which = @chain df_expand begin
        subset([:n_seeded_agents] => ByRow(==(n_seeded_agents_range[end])))
        select(Not(:policy))
        transform(
            [:similarity, :ob, :ib, :og, :ig] =>
                ByRow((s, xs...) -> findmax(==(s), xs)[2]) => :which_worst,
        )
        select(Not([:ob, :ib, :og, :ig]))
    end
    df_min = combine(groupby(df_which, [:generation, :seed])) do sdf
        x, i = findmin(sdf.similarity)
        (which_worst=sdf.which_worst[i], similarity_minimum=x)
    end
    df_really_min = combine(groupby(df_min, :generation)) do sdf
        x, i = findmin(sdf.similarity_minimum)
        (
            which_worst=sdf.which_worst[i],
            similarity_minimum=x,
            rgba=(cmap[sdf.which_worst[i]], 1.0),
        )
    end

    display(df_really_min)
    foreach(groupby(df_min, :seed)) do sdf
        transform!(sdf, :which_worst => ByRow(x -> (cmap[x], alpha)) => :rgb)
        lines!(ax, sdf.generation, sdf.similarity_minimum; color=sdf.rgb, label=nothing)
    end
    # df_margin_group = combine(
    #     groupby(df_min, [:n_seeded_agents, :generation]),
    #     :similarity_minimum => minimum => :similarity_minimum_group,
    # )
    # transform!(
    #     df_margin_group,
    #     :n_seeded_agents => ByRow(x -> cmap[cdict[x]]) => :rgb,
    #     # :agent_group => ByRow(x -> smap[x]) => :linestyle
    # )

    lines!(
        ax,
        df_really_min.generation,
        df_really_min.similarity_minimum;
        color=df_really_min.rgba,
    )
    # end
    begin
        color_lines = [
            LineElement(; color=cmap[i]) for (i, _) in enumerate(n_seeded_agents_range)
        ]
        transparency_markers = [
            MarkerElement(;
                color=(:black, alpha), marker=:rect, markersize=15, strokewidth=1
            ) for alpha in (1, 0.1)
        ]
        objects = [color_lines, transparency_markers]
        object_labels = [string.(n_seeded_agents_range), ["Over all runs", "This run"]]
        legend_titles = ["Number of\nAllD agents", "Which minimum?"]
        Legend(fig[1, 2], objects, object_labels, legend_titles)
    end
    for filetype in ("png", "pdf")
        # save("projects/aamas/figures/basin-of-attraction/margin-distance-$(norm).$filetype", fig)
    end
    display(fig)
end

# Look at which Q-value is causing the problem in each case, perhaps it's
# consistently one Q-value that gets "left behind".

# Not the case for 150

# df_plot = @chain df_policies begin
#     subset(:generation => ByRow(==(n_generations)))
#     sort!(:similarity)
#     groupby([:n_seeded_agents, :agent_group])
#     combine(sdf -> sdf[1, :])
#     select(Not(:similarity, :generation, :policy))
#     leftjoin(df_policies; on=[:n_seeded_agents, :seed, :agent_group, :agent])
# end

# @chain granular_output begin
#     groupby(:n_seeded_agents)
#     combine(:cooperation => x -> mean(>(0.5), x), :fairness => mean)
# end

# hist(
#     subset(granular_output, :n_seeded_agents => ByRow(==(50))).cooperation;
#     normalization=:probability,
# )

# df_bad_initial_values = @chain granular_output begin
#     subset(
#         # :n_seeded_agents => ByRow(==(50)),
#         :cooperation => ByRow(<(0.5)),
#     )
#     transform(:policies_per_generation => ByRow(first) => :initial_policies)
#     # select(:initial_policies)
# end

# df_good_initial_values = @chain granular_output begin
#     subset(
#         # :n_seeded_agents => ByRow(==(0)),
#         :cooperation => ByRow(>(0.5)),
#     )
#     transform(:policies_per_generation => ByRow(first) => :initial_policies)
#     # select(:initial_policies)
# end

# begin
#     bad_granular_output = DataFrame(
#         :n_seeded_agents => Int[],
#         :seed => Int[],
#         :cooperation => Float64[],
#         :fairness => Float64[],
#         :margin_distance => Vector{Float64}[],
#         # :final_policies => Vector{Policy}[],
#         :policies_per_generation => Vector{Vector{Policy}}[],
#     )

#     for (agent_policies, seed, n_seeded_agents) in
#         eachrow(df_bad_initial_values[:, [:initial_policies, :seed, :n_seeded_agents]])
#         @show seed
#         rng = Xoshiro(seed + population_size)
#         ideal_policy_maj, ideal_policy_min = 5 .* get_ideal_policies(norm; p, rng)
#         abm = initialise_rlabm_with_policies(
#             norm,
#             judge_characteristics,
#             agent_characteristics,
#             utilities,
#             global_simulation_variables;
#             rng,
#             agent_policies,
#         )
#         data = (;
#             df_cooperation=DataFrame(
#                 :donor => Int64[], :recipient => Int64[], :action => Bool[]
#             ),
#             policies=Vector{Policy}[],
#         )
#         train_and_collect!(abm, data)
#         coop_fairness = get_coop_fairness(abm, data)
#         coop_fairness_vec = coop_fairness
#         sum_of_margins = get_sum_of_margins_over_time(
#             abm, data; ideal_strategies=pol_to_i.((ideal_policy_maj, ideal_policy_min))
#         )
#         # final_policies = copy(abm.agent_data.policies)
#         policies_this_generation = data.policies
#         push!(
#             bad_granular_output,
#             (
#                 n_seeded_agents,
#                 seed,
#                 coop_fairness...,
#                 sum_of_margins,
#                 policies_this_generation,
#             ),
#         )
#     end
# end

# bad_granular_output

# @chain bad_granular_output begin
#     groupby(:n_seeded_agents)
#     combine(:cooperation => x -> mean(>(0.5), x), :fairness => mean)
# end

# hist(
#     subset(bad_granular_output, :n_seeded_agents => ByRow(==(25))).cooperation;
#     normalization=:probability,
# )

# begin
#     good_granular_output = DataFrame(
#         :n_seeded_agents => Int[],
#         :seed => Int[],
#         :cooperation => Float64[],
#         :fairness => Float64[],
#         :margin_distance => Vector{Float64}[],
#         # :final_policies => Vector{Policy}[],
#         :policies_per_generation => Vector{Vector{Policy}}[],
#     )

#     for (agent_policies, seed, n_seeded_agents) in
#         eachrow(df_good_initial_values[:, [:initial_policies, :seed, :n_seeded_agents]])
#         @show seed
#         rng = Xoshiro(seed + population_size)
#         ideal_policy_maj, ideal_policy_min = 5 .* get_ideal_policies(norm; p, rng)
#         abm = initialise_rlabm_with_policies(
#             norm,
#             judge_characteristics,
#             agent_characteristics,
#             utilities,
#             global_simulation_variables;
#             rng,
#             agent_policies,
#         )
#         data = (;
#             df_cooperation=DataFrame(
#                 :donor => Int64[], :recipient => Int64[], :action => Bool[]
#             ),
#             policies=Vector{Policy}[],
#         )
#         train_and_collect!(abm, data)
#         coop_fairness = get_coop_fairness(abm, data)
#         coop_fairness_vec = coop_fairness
#         sum_of_margins = get_sum_of_margins_over_time(
#             abm, data; ideal_strategies=pol_to_i.((ideal_policy_maj, ideal_policy_min))
#         )
#         # final_policies = copy(abm.agent_data.policies)
#         policies_this_generation = data.policies
#         push!(
#             good_granular_output,
#             (
#                 n_seeded_agents,
#                 seed,
#                 coop_fairness...,
#                 sum_of_margins,
#                 policies_this_generation,
#             ),
#         )
#     end
# end

# good_granular_output

# @chain good_granular_output begin
#     groupby(:n_seeded_agents)
#     combine(:cooperation => x -> mean(>(0.5), x), :fairness => mean)
# end

# hist(
#     subset(good_granular_output, :n_seeded_agents => ByRow(==(0))).cooperation;
#     normalization=:probability,
# )

# seed = 1
# n_seeded_agents = population_size
# rng = Xoshiro(seed)
# # To begin with all agents have a random policy policy
# ideal_policy_maj, ideal_policy_min = 5 .* get_ideal_policies(norm; p, rng)
# agent_policies = starting_policy_scale * rand(rng, Policy, population_size)
# # Then we replace some with AllD
# if n_seeded_agents != 0
#     seeded_agent_policies = [
#         starting_policy_scale * i_to_pol(15) for i in 1:n_seeded_agents
#     ]
#     seeded_agents = sample(rng, 1:population_size, n_seeded_agents; replace=false)
#     for (agent, seeded_policy) in zip(seeded_agents, seeded_agent_policies)
#         agent_policies[agent] = seeded_policy
#     end
# end
# abm = initialise_rlabm_with_policies(
#     norm,
#     judge_characteristics,
#     agent_characteristics,
#     utilities,
#     global_simulation_variables;
#     rng,
#     agent_policies,
# )

let
    fig = Figure(; resolution=(600, 600))
    ax = Axis(
        fig[1, 1];
        # xlabel="Generation",
        # ylabel="Distance from Q-value to margin",
        # title="Minimum L1 distance between Q-values and line x=y\nfor multiple runs: Norm-$norm",
    )
    cdict = Dict(n_seeded_agents_range .=> 1:length(n_seeded_agents_range))
    cmap = cgrad(:Egypt, 4; categorical=true)
    info_color_dict = Dict((:ob, :ib, :og, :ig) .=> cmap)
    smap = Dict(false => :dash, true => :solid)
    alpha = 0.1
    sdf = @chain df_policies begin
        subset(
            # :seed => ByRow(==(1)), 
            # :agent => ByRow(==(1)), 
            :n_seeded_agents => ByRow(==(50)),
        )
        groupby([:generation])
        combine(:policy => mean => :policy)
        transform(:policy => ByRow(pol_to_points) => AsTable)
    end
    display(sdf)
    for col in (:ob, :ib, :og, :ig)
        color = info_color_dict[col]
        lines!(ax, sdf[!, :generation], reduce.(-, sdf[!, col]); color)
    end
    # for col in (:ob, :ib, :og, :ig)
    #     color = info_color_dict[col]
    #     lines!(ax, sdf[!, :generation], first.(sdf[!, col]); color=:minority)
    # end
    # lines!(ax, sdf[!, :generation], first.(sdf[!, :ob]))
    display(fig)
end

@chain granular_output begin
    groupby(:n_seeded_agents)
    combine(:cooperation => mean)
end
