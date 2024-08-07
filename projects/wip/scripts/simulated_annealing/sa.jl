using Statistics
using Random
using Base: Fix1, Fix2

using StaticArrays

using IR

using CSV
using DataFrames
using StatsBase
using Tidier

whichmax(x, y) = argmax((x, y)) - 1

struct AgentGroups
    truemax::UInt8
    population_size::UInt8
end
Base.checkbounds(ag::AgentGroups, x) = 0 < x <= ag.population_size
Base.getindex(ag::AgentGroups, x) = x <= ag.truemax

struct AgentDataABM
    groups::AgentGroups
    strategies::Vector{Int8}
    utilities::Vector{Int}
    reputations::BitVector
    interacted_as_donor::BitVector
    n_donor_interactions::Vector{Int}
    n_cooperations::Vector{Int}
end

struct IRABM{A,P,R}
    agent_data::A
    properties::P
    rng::R
end

const CNorm = SArray{NTuple{4,2},Bool}
toCNorm(i::Integer) = CNorm(i >> shift & 1 != 0 for shift in 0:15)
toCNorm(n) = CNorm([n;;;; n])

# Allocate (mutable) agent data
function initialise_abm(
    norm,
    judge_characteristics,
    agent_characteristics,
    utilities,
    global_simulation_variables;
    rng,
)
    # Calculate some auxiliary properties based on inputs
    population_size,
    majority_proportion,
    n_generations,
    _,
    n_training_interactions,
    n_data_interactions,
    exploration_rate,
    strategy_range,
    _... = global_simulation_variables
    if norm > 255
        # @warn "Reading norm as 3rd order"
        norm_matrix = toCNorm(norm)
    else
        norm_matrix = toCNorm(iNorm(norm))
    end

    # (preallocate some randomness that would allocate otherwise)
    agents_to_interact_training = [
        SVector{2,Int}(rand(rng, 1:population_size, 2)) for _ in 1:n_training_interactions
    ]
    agents_to_interact_data = [
        SVector{2,Int}(rand(rng, 1:population_size, 2)) for _ in 1:n_data_interactions
    ]
    agents_to_update_and_compare = [
        SVector{2,Int}(rand(rng, 1:population_size, 2)) for _ in 1:n_generations
    ]
    update_thresholds = rand(rng, n_generations)
    update_by_explore = rand(rng, n_generations) .< exploration_rate
    preallocated_randomness = (;
        agents_to_interact_training,
        agents_to_interact_data,
        agents_to_update_and_compare,
        update_thresholds,
        update_by_explore,
    )
    properties = merge(
        (; norm, norm_matrix),
        judge_characteristics,
        agent_characteristics,
        global_simulation_variables,
        utilities,
        preallocated_randomness,
    )

    majority_population_size = floor(Int, majority_proportion * population_size)
    groups = AgentGroups(majority_population_size, population_size)
    strategies = rand(rng, strategy_range, population_size)
    utilities = zeros(Int, population_size)
    reputations = rand(rng, Bool, population_size)
    interacted_as_donor = falses(population_size)
    n_donor_interactions = zeros(population_size)
    n_cooperations = zeros(population_size)

    agent_data = AgentDataABM(
        groups,
        strategies,
        utilities,
        reputations,
        interacted_as_donor,
        n_donor_interactions,
        n_cooperations,
    )
    return IRABM(agent_data, properties, rng)
end

abm = let
    rng = Xoshiro(1)
    norm = 150

    # Magic constants
    strategy_range = 0:15

    # Global simulation variables
    Z = population_size = 50
    majority_proportion = 0.9
    n_generations = population_size * 100
    generation_length = 10 * population_size
    n_training_interactions = n_generations * generation_length
    n_data_interactions = n_training_interactions
    μ = exploration_rate = 1 / 10 # 1 / population_size
    τ = update_reputation_probability = 0.9
    β = selection_intensity = 1

    global_simulation_variables = (;
        population_size,
        majority_proportion,
        n_generations,
        generation_length,
        n_training_interactions,
        n_data_interactions,
        exploration_rate,
        strategy_range,
        update_reputation_probability,
        selection_intensity,
    )

    # Characteristics of players (rate of errors)
    majority_α = SA[0.00, 0.00]
    minority_α = SA[0.00, 0.00]
    majority_ε = 0.01
    minority_ε = 0.01
    agent_characteristics = (; majority_α, minority_α, majority_ε, minority_ε)

    # Characteristics of judge and norm used
    judge_α = SA[0.00, 0.00, 0.00, 0.00]
    judge_ε = 0.01
    judge_characteristics = (; judge_α, judge_ε)

    # The costs and benefits of interacting
    majority_benefit = 10
    minority_benefit = 10
    majority_cost = 1
    minority_cost = 1
    utilities = (; majority_benefit, majority_cost, minority_benefit, minority_cost)
    initialise_abm(
        norm,
        judge_characteristics,
        agent_characteristics,
        utilities,
        global_simulation_variables;
        rng=rng,
    )
end

function act(X, info, abm::IRABM{AgentDataABM})
    agent_data = abm.agent_data
    properties = abm.properties
    rng = abm.rng
    α = agent_data.groups[X] ? properties.majority_α : properties.minority_α
    perceived_info = mistake(α, info) .> rand(rng, SVector{2,Float64})
    ε = agent_data.groups[X] ? properties.majority_ε : properties.minority_ε
    strategy = iStrategy(agent_data.strategies[X])
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
        mistake(properties.judge_α, judge_info) .> rand(rng, SVector{4,Float64}) # preallocate?
    prob_good = (Fix1(mistake, properties.judge_ε) ∘ Fix1(lerp, properties.norm_matrix))(
        judge_perceived_info
    )
    judgement = prob_good > rand(rng) # preallocate?
    return judgement
end

function learn((X, Y), update_step, abm::IRABM{AgentDataABM})
    agent_data = abm.agent_data
    properties = abm.properties
    rng = abm.rng
    avg_interactions_per_agent = properties.generation_length / properties.population_size
    if properties.update_by_explore[update_step]
        new_strategy = rand(rng, properties.strategy_range)
    else
        utility_delta = (agent_data.utilities[Y] - agent_data.utilities[X])
        normalised_utility_delta = utility_delta / avg_interactions_per_agent
        update_strategy_probability = inv(
            1 + exp(-properties.selection_intensity * normalised_utility_delta)
        )
        update_strategy =
            update_strategy_probability > properties.update_thresholds[update_step]
        if update_strategy
            new_strategy = agent_data.strategies[Y]
        else
            new_strategy = agent_data.strategies[X]
        end#if
    end#if
    return new_strategy
end

function train!(abm::IRABM{AgentDataABM})
    agent_data = abm.agent_data
    properties = abm.properties
    rng = abm.rng
    update_step = 0
    for interaction_number in 1:(properties.n_training_interactions)
        X, Y = properties.agents_to_interact_training[interaction_number]
        # Determine the action taken by the agent
        is_same_group = agent_data.groups[X] == agent_data.groups[Y]
        donor_is_good = agent_data.reputations[X]
        recipient_is_good = agent_data.reputations[Y]
        info = SA[is_same_group, recipient_is_good]
        action, perceived_info = act(X, info, abm)
        agent_data.interacted_as_donor[X] = true # Update that the chosen donor has now been a donor
        # Possibly update donor's reputation via judgement
        update_reputation = rand(rng) < properties.update_reputation_probability # τ
        if update_reputation
            judge_info = SA[is_same_group, recipient_is_good, action, donor_is_good]
            judgement = judge(judge_info, abm)
            agent_data.reputations[X] = judgement
        end
        # Deal with utilities
        cost = agent_data.groups[X] ? properties.majority_cost : properties.minority_cost
        benefit =
            agent_data.groups[Y] ? properties.majority_benefit : properties.minority_benefit
        agent_data.utilities[X] -= cost * action
        agent_data.utilities[Y] += benefit * action
        # Learn
        if mod(interaction_number, properties.generation_length) == 1
            update_step += 1
            A, B = properties.agents_to_update_and_compare[update_step]
            new_strategy = learn((A, B), update_step, abm)
            agent_data.strategies[A] = new_strategy
            agent_data.utilities .= 0
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
        donor_is_good = agent_data.reputations[X]
        recipient_is_good = agent_data.reputations[Y]
        info = SA[is_same_group, recipient_is_good]
        action, _ = act(X, info, abm)
        agent_data.n_donor_interactions[X] += 1
        agent_data.n_cooperations[X] += action
        update_reputation = rand(rng) < properties.update_reputation_probability # τ
        if update_reputation
            judge_info = SA[is_same_group, recipient_is_good, action, donor_is_good]
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

function run!(abm)
    train!(abm)
    collect_data!(abm)
    return nothing
end

run!(abm)

function process_data(abm)
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

# process_data(abm)

## Multirun stuff
# begin
#     n_runs = 50
#     output = DataFrame(:norm => Int[], :cooperation => Float64[], :fairness => Float64[])
#     granular_output = DataFrame(
#         :norm => Int[],
#         :run => Int64[],
#         :cooperation => Float64[],
#         :fairness => Float64[],
#         :strategies => Vector{Int}[],
#     )
#     for norm in (0, 150, 192, 195, 243)
#         rng = Xoshiro(2)
#         # norm = 195

#         # Magic constants
#         strategy_range = 0:15

#         # Global simulation variables
#         Z = population_size = 50
#         majority_proportion = 0.9
#         n_generations = population_size * 20
#         generation_length = 10 * population_size
#         n_training_interactions = n_generations * generation_length
#         n_data_interactions = n_training_interactions
#         μ = exploration_rate = 1 / population_size
#         τ = update_reputation_probability = 0.9
#         β = selection_intensity = 1

#         global_simulation_variables = (;
#             population_size,
#             majority_proportion,
#             n_generations,
#             generation_length,
#             n_training_interactions,
#             n_data_interactions,
#             exploration_rate,
#             strategy_range,
#             update_reputation_probability,
#             selection_intensity,
#         )

#         # Characteristics of players (rate of errors)
#         majority_α = SA[0.00, 0.00]
#         minority_α = SA[0.00, 0.00]
#         majority_ε = 0.01
#         minority_ε = 0.01
#         agent_characteristics = (; majority_α, minority_α, majority_ε, minority_ε)

#         # Characteristics of judge and norm used
#         judge_α = SA[0.00, 0.00, 0.00, 0.0]
#         judge_ε = 0.01
#         judge_characteristics = (; judge_α, judge_ε)

#         # The costs and benefits of interacting
#         majority_benefit = 10
#         minority_benefit = 10
#         majority_cost = 1
#         minority_cost = 1
#         utilities = (; majority_benefit, majority_cost, minority_benefit, minority_cost)
#         # seeds = rand(UInt16, n_runs)
#         seeds = 1:n_runs
#         coop_fairness_vec = zeros(SVector{2,Float64}, n_runs)
#         @show norm
#         for (i_seed, seed) in enumerate(seeds)
#             # print(i_seed)
#             rng = Xoshiro(seed)
#             abm = initialise_abm(
#                 norm,
#                 judge_characteristics,
#                 agent_characteristics,
#                 utilities,
#                 global_simulation_variables;
#                 rng=rng,
#             )
#             run!(abm)
#             coop_fairness_vec[i_seed] = process_data(abm)
#             coop, fairness = process_data(abm)
#             println(
#                 "run: $i_seed, ($(round(coop; digits=3)), $(round(fairness; digits=3)))"
#             )
#             # display(countmap(abm.agent_data.strategies))
#             push!(
#                 granular_output, (norm, i_seed, coop, fairness, abm.agent_data.strategies)
#             )
#         end
#         mean_cooperation, mean_fairness = mean(
#             reinterpret(reshape, Float64, coop_fairness_vec); dims=2
#         )
#         println(
#             "$norm: $(round(mean_cooperation; sigdigits=3)), $(round(mean_fairness; sigdigits=3))",
#         )
#         push!(output, (norm, mean_cooperation, mean_fairness))
#     end
#     output
# end

# TODO: We have "3th order + group" norms working, now we need to write
# evolutionary algorithms that can add and remove rules based on performance as
# well as doing temperature-based rule updates. To do this we need to be able to
# convert `CNorm`s to rule-lists and vice-versa.

# We need to have rule-lists if we want "do not update reputation" to be a rule.

struct Rule
    input::BitVector
    output::Bool
end

# Good enough for government work
function f_rec(cn, _indices...; rlist=Rule[])
    indices = (_indices..., :, :, :, :)[1:4]
    x = first(getindex(cn, indices...))
    if allequal(view(cn, indices...))
        push!(rlist, Rule([(_indices .- 1)...], x))
    else
        return Rule[f_rec(cn, _indices..., 1)..., f_rec(cn, _indices..., 2)...]
    end
    return rlist
end

cn = reshape([1, 0, 1, 1, 0, 0, 0, 1], (2, 2, 2))

CNormtoList(cn::CNorm) = f_rec(cn)

CNormtoList(toCNorm(iNorm(192)))

function ListtoCNorm(list::Vector{Rule})
    cnorm = zeros(Bool, 2, 2, 2, 2)
    @views cnorm[:, :, :, 2] .= 1
    for rule in list
        idx = rule.input .+ 1
        @views cnorm[idx..., :, :, :, :] .= rule.output
    end
    return CNorm(cnorm)
end

function add_rule(cn::CNorm, new_rule::Rule)
    editable_cn = MArray(copy(cn))
    editable_cn[(new_rule.input .+ 1)..., :, :, :, :] .= new_rule.output
    return CNorm(editable_cn)
end

function remove_rule(cn::CNorm, input::AbstractVector)
    list = CNormtoList(cn)
    i = findfirst(r -> r.input == input, list)
    if isnothing(i)
        @warn "input $input not found in norm"
        return cn
    end
    deleteat!(list, i)
    return ListtoCNorm(list)
end

function multisim(
    norm;
    n_runs=10,
    judge_characteristics,
    agent_characteristics,
    utilities,
    global_simulation_variables,
    rng,
)
    seeds = rand(rng, UInt, n_runs)
    coop_fairness_vec = zeros(SVector{2,Float64}, n_runs)
    # @show norm
    for (i_seed, seed) in enumerate(seeds)
        # print(i_seed)
        rng = Xoshiro(seed)
        abm = initialise_abm(
            norm,
            judge_characteristics,
            agent_characteristics,
            utilities,
            global_simulation_variables;
            rng=rng,
        )
        run!(abm)
        coop_fairness_vec[i_seed] = process_data(abm)
        coop, fairness = process_data(abm)
        # println("run: $i_seed, ($(round(coop; digits=3)), $(round(fairness; digits=3)))")
        # push!(granular_output, (norm, i_seed, coop, fairness, abm.agent_data.strategies))
    end
    mean_cooperation, mean_fairness = mean(
        reinterpret(reshape, Float64, coop_fairness_vec); dims=2
    )
    return mean_cooperation, mean_fairness
end

function opt_f(norm, rng)
    # Magic constants
    strategy_range = 0:15
    norm = @evalpoly(2, norm...)

    # Global simulation variables
    Z = population_size = 50
    majority_proportion = 0.9
    n_generations = population_size * 100
    generation_length = 10 * population_size
    n_training_interactions = n_generations * generation_length
    n_data_interactions = n_training_interactions
    μ = exploration_rate = 1 / 10 # 1 / population_size
    τ = update_reputation_probability = 0.9
    β = selection_intensity = 1

    global_simulation_variables = (;
        population_size,
        majority_proportion,
        n_generations,
        generation_length,
        n_training_interactions,
        n_data_interactions,
        exploration_rate,
        strategy_range,
        update_reputation_probability,
        selection_intensity,
    )

    # Characteristics of players (rate of errors)
    majority_α = SA[0.00, 0.00]
    minority_α = SA[0.00, 0.00]
    majority_ε = 0.01
    minority_ε = 0.01
    agent_characteristics = (; majority_α, minority_α, majority_ε, minority_ε)

    # Characteristics of judge and norm used
    judge_α = SA[0.00, 0.00, 0.00, 0.00]
    judge_ε = 0.01
    judge_characteristics = (; judge_α, judge_ε)

    # The costs and benefits of interacting
    majority_benefit = 10
    minority_benefit = 10
    majority_cost = 1
    minority_cost = 1
    utilities = (; majority_benefit, majority_cost, minority_benefit, minority_cost)
    c, f = multisim(
        norm;
        n_runs=50,
        judge_characteristics,
        agent_characteristics,
        utilities,
        global_simulation_variables,
        rng,
    )
    return cost(c, f)
end

function cost(coop, fairness)
    return 3 - (2 * coop + fairness)
end

random_neighbouring_norm(cn::CNorm) = random_neighbouring_norm(Random.default_rng(), cn)
function random_neighbouring_norm(rng, cn::CNorm)
    add_or_remove = rand(rng, Bool)
    # @show add_or_remove == new_input[1]
    if add_or_remove
        new_input..., new_output = rand(rng, Bool, 4)
        # println("adding $new_input")
        return add_rule(cn, Rule(new_input, new_output))
    else
        list = CNormtoList(cn)
        # @show length(list)
        i = rand(rng, 1:length(list))
        deleteat!(list, i)
        return ListtoCNorm(list)
    end
end

temperature(r) = r

# Simulated Annealing:
rng = Xoshiro(0)
# s0 = toCNorm(iNorm(195))
s0 = rand(rng, CNorm)
e0 = opt_f(s0, rng)
kmax = 1_000
s = s0
e = e0
smin = s
emin = e
for k in 1:kmax
    # @show k
    T = temperature(1 - (k + 1) / kmax)
    snew = random_neighbouring_norm(s)
    enew = opt_f(snew, rng)
    if (enew < emin)
        emin = enew
        smin = snew
    end
    if (enew < e) || (rand(rng) > exp(-(enew - e) / T))
        s = snew
        e = enew
        # println(enew)
        print(k)
    end
end

sbest = toCNorm(iNorm(195))
ebest = opt_f(sbest, rng)
smin
emin
