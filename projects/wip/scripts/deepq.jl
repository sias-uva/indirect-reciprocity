using Statistics
using Random
using Base: Fix1, Fix2

using Flux
using StaticArrays
using StatsBase

using IR
import IR: iNorm, iStrategy
using IRUtils

# const Norm = SArray{NTuple{2,2},Bool}
const Strategy = SArray{NTuple{1,2},Bool} # Alias
# IR.iNorm(i::Integer) = Norm(i >> shift & 1 != 0 for shift in 0:3)
IR.iStrategy(i::Integer) = Strategy(i >> shift & 1 != 0 for shift in 0:1)

whichmax(x, y) = argmax((x, y)) - 1

struct AgentGroups
    truemax::UInt8
    population_size::UInt8
end
Base.checkbounds(ag::AgentGroups, x) = 0 < x <= ag.population_size
Base.getindex(ag::AgentGroups, x) = x <= ag.truemax

struct AgentDataABM
    strategies::Vector{Strategy}
    utilities::Vector{Int}
    reputations::BitVector
    interacted_as_donor::BitVector
    n_donor_interactions::Vector{Int}
    n_cooperations::Vector{Int}
end

struct IRABM{A,P}
    agent_data::A
    properties::P
end

function initialise_abm(
    norm, # Flux model
    agent_characteristics,
    utilities,
    global_simulation_variables,
)
    # Calculate some auxiliary properties based on inputs
    (; population_size) = global_simulation_variables
    # norm_matrix = iNorm(norm)

    properties = merge(
        (; norm), agent_characteristics, global_simulation_variables, utilities
    )
    strategies = rand(Strategy, population_size)
    utilities = zeros(Int, population_size)
    reputations = rand(Bool, population_size)
    interacted_as_donor = falses(population_size)
    n_donor_interactions = zeros(population_size)
    n_cooperations = zeros(population_size)

    agent_data = AgentDataABM(
        strategies,
        utilities,
        reputations,
        interacted_as_donor,
        n_donor_interactions,
        n_cooperations,
    )
    return IRABM(agent_data, properties)
end

Random.seed!(1)
abm = let
    norm = Chain(Dense(2 => 4), Dense(4 => 4), Dense(4 => 2), softmax)

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
        update_reputation_probability,
        selection_intensity,
    )

    # Characteristics of players (rate of errors)
    majority_α = SA[0.00, 0.00]
    minority_α = SA[0.00, 0.00]
    majority_ε = 0.01
    minority_ε = 0.01
    agent_characteristics = (; majority_α, minority_α, majority_ε, minority_ε)

    # The costs and benefits of interacting
    majority_benefit = 10
    minority_benefit = 10
    majority_cost = 1
    minority_cost = 1
    utilities = (; majority_benefit, majority_cost, minority_benefit, minority_cost)
    initialise_abm(norm, agent_characteristics, utilities, global_simulation_variables)
end

loss()
