using Statistics
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

# Figure 5
begin
    granular_metric_output = DataFrame(
        :seed => Int[], :norm => Int[], :cooperation => Float64[], :fairness => Float64[]
    )
    loop_norms = 0:255
    for norm in loop_norms
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
        τ = update_reputation_probability = 1
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
        n_runs = 50
        seeds = 1:n_runs
        coop_fairness_vec = zeros(SVector{2,Float64}, n_runs)
        for (i_seed, seed) in enumerate(seeds)
            rng = Xoshiro(seed)
            abm = initialise_rlabm(
                norm,
                judge_characteristics,
                agent_characteristics,
                utilities,
                global_simulation_variables;
                rng=rng,
            )
            run!(abm)
            coop_fairness = get_coop_fairness(abm)
            coop_fairness_vec[i_seed] = coop_fairness
            push!(granular_metric_output, (seed, norm, coop_fairness...))
        end
    end
    CSV.write(
        "projects/ijcai2024/data/figure5/granular_rl_data_$(abm.properties.majority_benefit).csv",
        granular_metric_output,
    )
end

# Figure 6
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

for benefit in 3:9
    norm = 195
    # @show norm
    (
        global_simulation_variables,
        agent_characteristics,
        judge_characteristics,
        utilities,
        p,
    ) = let # Set ABM parameters
        # norm = 243
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
        τ = update_reputation_probability = 1.0
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
        majority_benefit = benefit
        minority_benefit = benefit
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
        (
            global_simulation_variables,
            agent_characteristics,
            judge_characteristics,
            utilities,
            p,
        )
    end

    function generate_granular_metric_output(
        norm;
        global_simulation_variables,
        agent_characteristics,
        judge_characteristics,
        utilities,
        p,
    )
        starting_policy_scale = 3
        population_size = global_simulation_variables.population_size
        granular_output = DataFrame(
            :n_seeded_agents => Int[],
            :seed => Int[],
            :cooperation => Float64[],
            :fairness => Float64[],
            :policies_per_generation => Vector{Vector{Policy}}[],
        )
        n_seeded_agents_range = 0:2:population_size
        n_runs = 50
        for n_seeded_agents in n_seeded_agents_range
            seeds = 1:n_runs
            coop_fairness_vec = zeros(SVector{2,Float64}, n_runs)
            @show n_seeded_agents
            for (i_seed, seed) in enumerate(seeds)
                @show i_seed
                rng = Xoshiro(seed)
                # To begin with all agents have a random policy policy
                ideal_policy_maj, ideal_policy_min =
                    starting_policy_scale .* get_ideal_policies(norm; p, rng)
                agent_policies = starting_policy_scale * rand(rng, Policy, population_size)
                # Then we replace some with the ideal policy
                if n_seeded_agents != 0
                    seeded_agents = sample(
                        rng, 1:population_size, n_seeded_agents; replace=false
                    )
                    for agent in seeded_agents
                        gsv = global_simulation_variables
                        if agent > round(Int, gsv.population_size * gsv.majority_proportion)
                            seeded_policy = ideal_policy_min
                        else
                            seeded_policy = ideal_policy_maj
                        end
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
                policies_this_generation = data.policies
                push!(
                    granular_output,
                    (n_seeded_agents, seed, coop_fairness..., policies_this_generation),
                )
            end
        end
        return granular_output
    end

    granular_metric_output = generate_granular_metric_output(
        norm;
        global_simulation_variables,
        agent_characteristics,
        judge_characteristics,
        utilities,
        p,
    )

    CSV.write("projects/ijcai2024/data/fig_appendix_c1/$(norm)", granular_metric_output)

    dfcoop = @chain granular_metric_output begin
        groupby([:n_seeded_agents])
        combine(:cooperation => mean, :fairness => mean)
    end
    CSV.write("projects/ijcai2024/data/figure6/$(norm)_cooperation_$benefit.csv", dfcoop)
end
