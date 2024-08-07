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

# function get_strategy_distribution(abm)
#     abm.agent_data.policies
# end

abm = let
    rng = Xoshiro(3)
    norm = 243

    # Magic constants
    strategy_range = 0:15

    # Global simulation variables
    Z = population_size = 50
    majority_proportion = 0.9
    generations = population_size * 100
    generation_length = 5 * population_size
    n_training_interactions = generations * generation_length
    n_data_interactions = n_training_interactions
    μ = exploration_rate = 1 / 10 # 1 / population_size
    τ = update_reputation_probability = 0.5
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
    initialise_rlabm(
        norm,
        judge_characteristics,
        agent_characteristics,
        utilities,
        global_simulation_variables;
        rng=rng,
    )
end

data = (;
    strategies=(; minority_strategies=Vector{Int64}[], majority_strategies=Vector{Int64}[]),
    df_cooperation=DataFrame(:donor => Int64[], :recipient => Int64[], :action => Bool[]),
)
train_and_collect!(abm, data)
@chain data.df_cooperation begin
    transform(
        [:donor, :recipient] .=>
            ByRow(Fix1(getindex, abm.agent_data.groups)) .=> [:donor, :recipient],
    )
    groupby([:donor, :recipient])
    combine(nrow => :tot, :action => count => :coop)
    select(:donor, :recipient, [:coop, :tot] => ByRow((x, y) -> x / y) => :coop)
end

strategy_counts_per_generation = let
    iter = enumerate((data.strategies.minority_strategies, data.strategies.majority_strategies))
    mapreduce(vcat, iter) do (group, strategy_counts)
        df = DataFrame("generation" => Int64[], (string.(0:15) .=> Ref(Int64[]))...)
        for (i, vec) in enumerate(strategy_counts)
            values = [count(==(s), vec) for s in 0:15]
            push!(df, (i, values...))
        end
        df = stack(df, Not(:generation); variable_name=:strategy, value_name=:count)
        df.group .= group
        df
    end
end

category_counts_per_generation = let
    df = transform(
        strategy_counts_per_generation,
        :strategy => ByRow(categorise_strategy ∘ Fix1(parse, Int)) => :category,
    )
    combine(groupby(df, [:generation, :category]), :count => sum)
end

let
    data = category_counts_per_generation
    fig = Figure(; resolution=(600, 300))
    cmap = :viridis
    yticknames = ["Always defect", "Discriminatory", "Group-agnostic"]
    ax = Axis(
        fig[1, 1];
        xlabel="Generation",
        ylabel="Strategy",
        title="Majority group",
        yticks=0:2,
        ytickformat=x -> getindex.(Ref(yticknames), Int.(x .+ 1)),
    )
    heatmap!(
        ax,
        data.generation,
        data.category,
        data.count_sum;
        colormap=cmap,
        colorrange=(0, abm.properties.population_size),
    )
    Colorbar(fig[1, 2]; limits=(0, 1), colormap=cmap, vertical=true, label="Prevalence")
    for filetype in ("png", "pdf")
        # save("figures/abm/prevalence_heatmap_$norm.$filetype", fig)
    end
    return fig
end

get_coop_fairness(abm)
@time run!(abm)

## Multirun stuff
begin
    output = DataFrame(:norm => Int[], :cooperation => Float64[], :fairness => Float64[])
    granular_output = DataFrame(
        :seed => Int[], :norm => Int[], :cooperation => Float64[], :fairness => Float64[]
    )
    loop_norms = (0, 150, 192, 195, 243)# 0:255
    for norm in loop_norms#(0, 150, 192, 195, 243)
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
            push!(granular_output, (seed, norm, coop_fairness...))
        end
        mean_cooperation, mean_fairness = mean(
            reinterpret(reshape, Float64, coop_fairness_vec); dims=2
        )
        println(
            "$norm: $(round(mean_cooperation; sigdigits=3)), $(round(mean_fairness; sigdigits=3))",
        )
        push!(output, (norm, mean_cooperation, mean_fairness))
    end
    CSV.write(
        "projects/aamas/data/rl_data_$(abm.properties.majority_benefit)_$loop_norms.csv",
        output,
    )
    CSV.write(
        "projects/aamas/data/granular_rl_data_$(abm.properties.majority_benefit).csv",
        granular_output,
    )
end

# Animation stuff
let
    begin
        fig = Figure(; resolution=(600, 600))
        ax = Axis(
            fig[1, 1];
            aspect=DataAspect(),
            xlabel="Q-value defect",
            ylabel="Q-value cooperate",
        )
        limits!(ax, (0, 10), (0, 10))
        points = map(1:4) do i
            Observable(Point2f.(getindex.(abm.agent_data.policies, Ref([i, i + 4]))))
        end
        for i in 1:4
            scatter!(ax, points[i])
        end
        fig
    end

    abm.properties.n_training_interactions
    generation_length = abm.properties.n_training_interactions ÷ 1000
    generations = (1:(abm.properties.n_training_interactions ÷ generation_length)) .- 1

    record(
        fig, "projects/wip/figures/q_values.mp4", generations; framerate=60
    ) do generation
        agent_data = abm.agent_data
        properties = abm.properties
        rng = abm.rng
        for interaction_number in
            range(generation * generation_length + 1; length=generation_length)
            X, Y = properties.agents_to_interact_training[interaction_number]
            # Determine the action taken by the agent
            is_same_group = agent_data.groups[X] == agent_data.groups[Y]
            is_good = agent_data.reputations[Y]
            info = SA[is_same_group, is_good]
            action, perceived_info = act(X, info, abm)
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
            cost =
                agent_data.groups[X] ? properties.majority_cost : properties.minority_cost
            benefit = if agent_data.groups[Y]
                properties.majority_benefit
            else
                properties.minority_benefit
            end
            for (A, utility) in zip((X, Y), (action * -cost, action * benefit))
                new_policy = learn(A, utility, abm)
                agent_data.policies[A] = new_policy
                agent_data.utilities[A] += utility
            end
        end
        for i in 1:4
            points[i][] = Point2f.(getindex.(abm.agent_data.policies, Ref([i, i + 4])))
        end
    end
end
