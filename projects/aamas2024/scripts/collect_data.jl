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

##% Construct output containers
output = DataFrame(
    :seed => Int[], :norm => Int[], :cooperation => Float64[], :fairness => Float64[]
)

##% Which norms to loop over
loop_norms = [195, 243, 192, 210, 209] # For all norms, use 0:255

##% Perform loop
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
        push!(output, (seed, norm, coop_fairness...))
    end
end

CSV.write("projects/aamas2024/data/rl_data.csv", output)
