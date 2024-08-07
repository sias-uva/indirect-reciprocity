using Statistics
using Random
using Base: Fix1, Fix2
using CSV
using StaticArrays
using DataFrames
using CairoMakie
using IR

whichmax(x, y) = argmax((x, y)) - 1

function pol_to_i(policy)
    sa = SArray{NTuple{2,2},Int64,2,4}(reduce(whichmax, policy; dims=3))
    @evalpoly(2, sa...)
end

function IR.iStrategy(policy::SArray{Tuple{2,2,2},Float64,3,8})
    return sa = SArray{NTuple{2,2},Int64,2,4}(reduce(whichmax, policy; dims=3))
end

struct AgentGroups
    truemax::UInt8
    population_size::UInt8
end
Base.checkbounds(ag::AgentGroups, x) = 0 < x <= ag.population_size
Base.getindex(ag::AgentGroups, x) = x <= ag.truemax
Base.length(ag::AgentGroups) = ag.population_size
function Base.iterate(ag::AgentGroups, state=1)
    state > ag.population_size && return nothing
    state > ag.truemax && return (false, state + 1)
    return (true, state + 1)
end

struct AgentDataRL
    groups::AgentGroups
    policies::Vector{SArray{Tuple{2,2,2},Float64,3,8}}
    utilities::Vector{Float64}
    reputations::BitVector
    memories::Vector{SVector{3,Bool}}
    interacted_as_donor::BitVector
    n_donor_interactions::Vector{Int}
    n_cooperations::Vector{Int}
end

struct IRABM{A,P,R}
    agent_data::A
    properties::P
    rng::R
end

# Allocate (mutable) agent data
function initialise_rlabm(
    norm,
    judge_characteristics,
    agent_characteristics,
    utilities,
    global_simulation_variables;
    rng,
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
    agent_policies = rand(rng, SArray{Tuple{2,2,2},Float64,3,8}, population_size)
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

function act(X, info, abm::IRABM{AgentDataRL})
    agent_data = abm.agent_data
    properties = abm.properties
    rng = abm.rng
    α = agent_data.groups[X] ? properties.majority_α : properties.minority_α
    perceived_info = mistake(α, info) .> rand(rng, SVector{2,Float64})
    ε = agent_data.groups[X] ? properties.majority_ε : properties.minority_ε
    policy = agent_data.policies[X]
    strategy = SArray{NTuple{2,2},Int64,2,4}(reduce(whichmax, policy; dims=3))
    if properties.exploration_rate > rand(rng)
        prob_coop = 0.5
        action = rand(rng, Bool) # Explore
    else
        prob_coop = (Fix1(execution_oopsie, ε) ∘ Fix1(lerp, strategy))(perceived_info)
        action = prob_coop > rand(rng) # Q-learning
    end
    return action, perceived_info
end

function judge(judge_info, abm)
    properties = abm.properties
    rng = abm.rng
    judge_perceived_info =
        mistake(properties.judge_α, judge_info) .> rand(rng, SVector{3,Float64}) # preallocate?
    prob_good = (Fix1(mistake, properties.judge_ε) ∘ Fix1(lerp, properties.norm_matrix))(
        judge_perceived_info
    )
    judgement = prob_good > rand(rng) # preallocate?
    return judgement
end

function learn(A, utility, abm::IRABM{AgentDataRL})
    agent_data = abm.agent_data
    properties = abm.properties
    old_policy = agent_data.policies[A]
    !agent_data.interacted_as_donor[A] && return old_policy
    interaction = agent_data.memories[A]
    # println("Agent $(agent.id) learned $utility from $(agent.memory)")
    idx = interaction .+ 1
    ϕ = properties.learning_rate # alias for mathematical brevity
    old_q_value = old_policy[idx...]
    new_q_value = (1 - ϕ) * old_q_value + ϕ * utility # Q-learning
    linear_idx = interaction[1] + 2 * interaction[2] + 4 * interaction[3] + 1
    return setindex(old_policy, new_q_value, linear_idx)
end

function train!(abm::IRABM{AgentDataRL})
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
        agent_data.n_donor_interactions[X] += 1
        agent_data.n_cooperations[X] += action
        agent_data.interacted_as_donor[X] = true # Update that the chosen donor has now been a donor

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
    end
end

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
        # TODO: Remove global magic. Currently keeping as is because simplicity
        # and not breaking working code. Deadlines innit.
        if mod(interaction_number, properties.generation_length) == 0
            majority_policies = @views agent_data.policies[1:(agent_data.groups.truemax)]
            minority_policies = @views agent_data.policies[(agent_data.groups.truemax + 1):end]
            minority_strategies = pol_to_i.(majority_policies)
            majority_strategies = pol_to_i.(minority_policies)
            push!(data.strategies.minority_strategies, minority_strategies)
            push!(data.strategies.majority_strategies, majority_strategies)
        end
    end
end

function collect_data!(abm)
    agent_data = abm.agent_data
    properties = abm.properties
    rng = abm.rng
    agent_data.utilities .= 0
    agent_data.n_donor_interactions .= 0
    agent_data.n_cooperations .= 0
    for interaction_number in 1:(properties.n_data_interactions)
        X, Y = properties.agents_to_interact_data[interaction_number]
        # Determine the action taken by the agent
        is_same_group = agent_data.groups[X] == agent_data.groups[Y]
        is_good = agent_data.reputations[Y]
        info = SA[is_same_group, is_good]
        action, _ = act(X, info, abm)
        agent_data.n_donor_interactions[X] += 1
        agent_data.n_cooperations[X] += action
        update_reputation = rand(rng) < properties.update_reputation_probability # τ
        if update_reputation
            judge_info = SA[is_same_group, is_good, action]
            judgement = judge(judge_info, abm)
            agent_data.reputations[X] = judgement
        end
        cost = agent_data.groups[X] ? properties.majority_cost : properties.minority_cost
        benefit =
            agent_data.groups[Y] ? properties.majority_benefit : properties.minority_benefit
        for (A, utility) in zip((X, Y), (action * -cost, action * benefit))
            agent_data.utilities[A] += utility
        end
    end
end

function run!(abm::IRABM{AgentDataRL})
    return train!(abm)
    # return collect_data!(abm)
end

function get_coop_fairness(abm)
    # Calculate cooperativeness: n_cooperations/n_interactions
    cooperativeness =
        sum(abm.agent_data.n_cooperations) / abm.properties.n_data_interactions
    # Calculate fairness
    utilities = abm.agent_data.utilities
    maj_cutoff = abm.agent_data.groups.truemax
    maj_utilities = mean(@views utilities[1:maj_cutoff])
    min_utilities = mean(@views utilities[(maj_cutoff + 1):end])
    lower, higher = extrema((maj_utilities, min_utilities))
    fairness = lower / higher
    return SA[cooperativeness, fairness]
end

function get_coop_fairness(abm, data)
    df_cooperation = data.df_cooperation
    cooperativeness = count(df_cooperation.action) / nrow(df_cooperation)
    # Calculate fairness
    utilities = abm.agent_data.utilities
    maj_cutoff = abm.agent_data.groups.truemax
    maj_utilities = mean(@views utilities[1:maj_cutoff])
    min_utilities = mean(@views utilities[(maj_cutoff + 1):end])
    lower, higher = extrema((maj_utilities, min_utilities))
    fairness = lower / higher
    return SA[cooperativeness, fairness]
end

function strategy_prevalence(abm, data)
    iter = enumerate((
        data.strategies.minority_strategies, data.strategies.majority_strategies
    ))
    df_long = mapreduce(vcat, iter) do (group, strategy_counts)
        df = DataFrame("generation" => Int64[], (string.(0:15) .=> Ref(Int64[]))...)
        for (i, vec) in enumerate(strategy_counts)
            values = [count(==(s), vec) for s in 0:15]
            push!(df, (i, values...))
        end
        df = stack(df, Not(:generation); variable_name=:strategy, value_name=:count)
        df.group .= group # Group is 1 or 2 for minority or majority
        df
    end
    # Calculate minority prevalence
    df_wide_min = @chain df_long begin
        subset(:group => ByRow(==(2)))
        unstack(:strategy, :count)
        select(3:18)
    end
    mat_counts_min = Matrix{Float64}(df_wide_min)
    minority_pop_size = abm.properties.population_size - abm.agent_data.groups.truemax
    mat_prevalence_min = mat_counts_min ./ minority_pop_size
    # Calculate majority prevalence
    df_wide_maj = @chain df_long begin
        subset(:group => ByRow(==(1)))
        unstack(:strategy, :count)
        select(3:18)
    end
    mat_counts_maj = Matrix{Float64}(df_wide_maj)
    majority_pop_size = abm.agent_data.groups.truemax
    mat_prevalence_maj = mat_counts_maj ./ majority_pop_size
    return mat_prevalence_min, mat_prevalence_maj
end

function get_ideal_policies(i; p, rng)
    df = find_ESS(p)
    sdf = subset(
        df,
        :norm => ByRow(==(i)),
        :is_ess,
        [:majority_strat, :minority_strat] => ByRow((x, y) -> !(x == y == 0)),
    )
    n_ess = nrow(sdf)
    if n_ess != 0
        which_ess = rand(rng, 1:n_ess)
        return i_to_pol.((sdf.majority_strat[which_ess], sdf.minority_strat[which_ess]))
    else
        return i_to_pol.((12, 12))
    end
end

strategy_names = Dict{Int,String}(0 => "AllD", 3 => "pDisc", 12 => "Disc", 15 => "AllC")
for i in 0:15
    get!(strategy_names, i) do
        string(i)
    end
end

"""
    i_to_pol(i::Integer)

Takes an integer and returns a `Policy` with 1s and 0s such that the policy
would be functionally equivalent to the strategy integer input when evaluated.
"""
function i_to_pol(i::Integer)
    pol = zeros(2, 2, 2)
    gen = (i >> shift & 1 != 0 for shift in 0:3)
    for (idx, i) in enumerate(gen)
        pol[idx] = 1 - i
        pol[idx + 4] = i
    end
    return Policy(pol)
end

function get_n_other_strategies_over_time(abm, data; ideal_strategies=(3, 12))
    minority_strategies = data.strategies.minority_strategies # isa Vector{Vector{Int}}
    majority_strategies = data.strategies.majority_strategies # isa Vector{Vector{Int}}
    n_generations = length(minority_strategies)
    # For each generation
    return map(1:n_generations) do generation
        # How many strategies aren't Disc or InvDisc in either the maj or min group?
        sum((
            minority_strategies[generation], majority_strategies[generation]
        )) do group_strategies
            count(!in(ideal_strategies), group_strategies)
        end
    end
end

function get_sum_of_margins_over_time(abm, data; ideal_strategies)
    vector_of_policies = data.policies
    groups = abm.agent_data.groups
    return map(vector_of_policies) do policies
        # How far away is each policy from the margin?
        majority_policies = policies[1:(groups.truemax)]
        minority_policies = policies[(groups.truemax + 1):end]
        group_iterator = zip((majority_policies, minority_policies), ideal_strategies)
        tot_dist = minimum(group_iterator) do (policies, group_ideal_strategies)
            minimum(policies) do policy
                # Maximum perpendicular distance from the ideal strategy
                dist = maximum(group_ideal_strategies) do strategy
                    strategy_matrix = iStrategy(strategy)
                    minimum(CartesianIndices(strategy_matrix)) do idx
                        # x-axis = defect, y-axis = cooperate
                        @views x, y = policy[idx, :]
                        # If y > x i.e. y-x > 0, then we cooperate.
                        intended = strategy_matrix[idx] ? 1 : -1
                        intended * (y - x)
                    end
                end
                dist
            end
        end
        # 4 is #entries in strategy
        tot_dist
    end
end

function similarity_score_q(q_pair, optimal)
    distance = q_pair[1] - q_pair[2]
    optimal_action_sign = optimal ? -1 : 1
    return optimal_action_sign * distance
end

function similarity_score(policy, strategy)
    minimum(CartesianIndices((1:2, 1:2))) do I
        @views q_pair = policy[I, :]
        @views optimal = strategy[I]
        similarity_score_q(q_pair, optimal)
    end
end

function similarity_to_optimal(policy, group; optimal_combinations)
    maximum(optimal_combinations) do optimal_combination
        strategy = optimal_combination[2 - group]
        istrategy = iStrategy(strategy)
        similarity_score(policy, istrategy)
    end
end

function get_optimal_strategy_combinations(norm; p)
    df = find_ESS(p)
    sdf = subset!(
        df,
        :norm => ByRow(==(norm)),
        :is_ess,
        [:majority_strat, :minority_strat] => ByRow((x, y) -> !(x == y == 0)),
    )
    if !isempty(sdf.majority_strat)
        return [(r, b) for (r, b) in zip(sdf.majority_strat, sdf.minority_strat)]
    else
        return [(12, 12)]
    end
end

function pol_to_points(pol)
    keys = (:ob, :ib, :og, :ig)
    values = (Point2(pol[I, :]...) for I in CartesianIndices((1:2, 1:2)))
    return (; zip(keys, values)...)
end
