using Base: Fix1
using Random

using IR
using StaticArrays
using DataFrames
using CairoMakie
using Tidier
using AlgebraOfGraphics
import AlgebraOfGraphics as aog
using PrettyTables

include("../misc.jl")
include("abm_functions.jl")

function pair_prevalence(df_strategy_count)
    gdf = groupby(select(df_strategy_count, Not(:prevalence)), [:update_step, :group])
    pair_prevalence_mat(df_min, df_maj) = df_min.group_prevalence * df_maj.group_prevalence'
    return mean(map(splat(pair_prevalence_mat), Iterators.partition(gdf, 2)))
end

# Set global RNG seed
Random.seed!(0)

function multisimulate(
    norm,
    judge_characteristics,
    agent_characteristics,
    utilities,
    global_simulation_variables;
    n_simulations,
    rng_seed,
)

    # Set seed again so it's the same every time
    Random.seed!(rng_seed)
    seeds = rand(UInt, n_simulations)
    metrics = (;
        cooperation_rate_per_group=DataFrame[],
        cooperation_rate=Float64[],
        reputation=Float64[],
        time_each_strategy=DataFrame[],
        fairness=Float64[],
        pair_prevalence=Matrix{Float64}[],
    )
    for (i, seed) in enumerate(seeds)
        # println("Run: $i")
        # Run simulation with seed
        metadata, df_strategy_count, df_reputation_count, df_interaction, df_judgement, df_action = simulate(
            norm,
            judge_characteristics,
            agent_characteristics,
            utilities,
            global_simulation_variables;
            rng_seed=seed,
            strategy_range,
        )
        # Calculate metrics
        acrg = average_cooperation_rate_group(
            df_interaction, df_action; metadata.agent_groups
        )
        acrg.run .= i
        acr = average_cooperation_rate(df_action)
        ar = average_reputation(df_reputation_count; population_size)
        ates = average_time_each_strategy(df_strategy_count)
        # display(ates)
        ates.run .= i
        fairness = average_fairness(df_interaction, df_action; metadata.agent_groups)
        pair_prevalence_mat = pair_prevalence(df_strategy_count)
        # Store results
        push!(metrics.cooperation_rate_per_group, acrg)
        push!(metrics.cooperation_rate, acr)
        push!(metrics.reputation, ar)
        push!(metrics.time_each_strategy, ates)
        push!(metrics.fairness, fairness)
        push!(metrics.pair_prevalence, pair_prevalence_mat)
    end
    return metrics
end

begin
    # norm = 19
    # Global variables
    strategy_names = Dict{Int,String}(0 => "AllD", 3 => "pDisc", 12 => "Disc", 15 => "AllC")
    for i in 0:15
        get!(strategy_names, i) do
            string(i)
        end
    end

    # Magic constants
    strategy_range = 0:15

    # Global simulation variables
    Z = population_size = 100
    majority_proportion = 0.9
    n_updates = population_size * 100 #100
    n_interactions_per_update = 5 * Z
    μ = exploration_rate = 1 / population_size
    τ = update_reputation_probability = 0.1
    β = selection_intensity = 1

    global_simulation_variables = (;
        population_size,
        majority_proportion,
        n_updates,
        n_interactions_per_update,
        exploration_rate,
        update_reputation_probability,
        selection_intensity,
        strategy_range,
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
end

for norm in (192,)#(150, 192, 195, 243)
    metrics = multisimulate(
        norm,
        judge_characteristics,
        agent_characteristics,
        utilities,
        global_simulation_variables;
        n_simulations=16,
        rng_seed=1,
    )

    begin # Process results
        # Average cooperation rate per group
        df_coop_group = vcat(metrics.cooperation_rate_per_group...)
        average_cooperation_group = @chain df_coop_group begin
            groupby(:donor_group)
            combine([:Minority, :Majority] .=> mean .=> [:Minority, :Majority])
        end
        # Average cooperation rate
        average_cooperation = mean(metrics.cooperation_rate)
        # Average reputation
        reputation = mean(metrics.reputation)
        # Average time each strategy
        df_time_strategy = vcat(metrics.time_each_strategy...)
        average_time_strategy = @chain df_time_strategy begin
            groupby(:strategy)
            combine([:minority, :majority] .=> mean .=> [:minority, :majority])
        end
        avg_fairness = mean(metrics.fairness)
        avg_pair_prevalence = mean(metrics.pair_prevalence)
        r = x -> round(x; sigdigits=3)
        println("Norm: $(norm)")
        println("Average cooperation: $(average_cooperation |> r)")
        println("Average fairness: $avg_fairness")
        println("xy-coord = ($average_cooperation, $avg_fairness)")
        println()
        println("Cooperation breakdown:")
        println("Minority -> Minority: $(average_cooperation_group.Minority[1] |> r) ")
        println("Minority -> Majority: $(average_cooperation_group.Minority[2] |> r)")
        println("Majority -> Minority: $(average_cooperation_group.Majority[1] |> r)")
        println("Majority -> Majority: $(average_cooperation_group.Majority[2] |> r)")

        println()
        println("Average reputation: $(reputation |> r)")
        println()
        # println("Average strategy prevalence:")
        # pretty_table(
        #     average_time_strategy;
        #     formatters=ft_printf("%.3f", [2, 3]),
        #     highlighters=(hl_gt(0.2)),
        # )
    end

    function strategy_prevalence_multirun_heatmap(df_time_strategy; strategy_names)
        fig = Figure(; resolution=(600, 500))
        cmap = :viridis
        ax = Axis(
            fig[1, 1];
            xlabel="Run",
            ylabel="Strategy",
            title="",
            yticks=0:15,
            ytickformat=x -> getindex.(Ref(strategy_names), Int.(x)),
            aspect=1,
        )
        # Group by generation
        df = select(df_time_strategy, [:strategy, :majority, :run])
        heatmap!(df.run, df.strategy, df.majority; colormap=cmap, colorrange=(0, 1))
        Colorbar(fig[1, 2]; limits=(0, 1), colormap=cmap, vertical=true, label="Prevalence")

        for filetype in ("png", "pdf")
            # save("figures/abm/multirun_prevalence_heatmap_$(norm)_$(majority_benefit).$filetype", fig)
            # save("$norm.$filetype", fig)
        end
        return fig
    end
    # strategy_prevalence_multirun_heatmap(df_time_strategy; strategy_names)
    display(pair_prevalence_heatmap(avg_pair_prevalence; strategy_names))
end

norm = 195
metadata, df_strategy_count, df_reputation_count, df_interaction, df_judgement, df_action = simulate(
    norm,
    judge_characteristics,
    agent_characteristics,
    utilities,
    global_simulation_variables;
    rng_seed=0,
    strategy_range,
)

function categorise_strategy(i)
    i == 0 && return 0 # AllD
    !is_fair(i) && return 1 # Discriminatory
    return 2 # Group-agnostic
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
    for filetype in ("png", "pdf")
        save("figures/abm/prevalence_heatmap_$norm.$filetype", fig)
    end
    return fig
end

strategy_prevalence_by_group_heatmap(df_strategy_count; strategy_names)

function pair_prevalence_heatmap(df_strategy_count; strategy_names)
    fig = Figure(; resolution=(600, 500))
    cmap = :viridis
    ax = Axis(
        fig[1, 1];
        xlabel="Majority Strategy",
        ylabel="Minority Strategy",
        title="",
        yticks=0:15,
        ytickformat=x -> getindex.(Ref(strategy_names), Int.(x)),
        xticks=0:15,
        xtickformat=x -> getindex.(Ref(strategy_names), Int.(x)),
        aspect=1,
    )
    # Group by generation
    df = DataFrame(pair_prevalence(df_strategy_count), string.(0:15))
    df.strategy_maj = 0:15
    df = stack(
        df, Not(:strategy_maj); variable_name=:strategy_min, value_name=:joint_prevalence
    )
    transform!(df, :strategy_min => ByRow(Fix1(parse, Int)) => :strategy_min)
    heatmap!(
        df.strategy_maj,
        df.strategy_min,
        df.joint_prevalence;
        colormap=cmap,
        colorrange=(0, 1),
    )
    Colorbar(fig[1, 2]; limits=(0, 1), colormap=cmap, vertical=true, label="Prevalence")
    for filetype in ("png", "pdf")
        save(
            "figures/abm/pair_prevalence_heatmap_$(norm)_$(majority_benefit).$filetype", fig
        )
        save("$norm.$filetype", fig)
    end
    return fig
end

function pair_prevalence_heatmap(prevalence_matrix::AbstractMatrix; strategy_names)
    fig = Figure(; resolution=(600, 500))
    cmap = :viridis
    ax = Axis(
        fig[1, 1];
        xlabel="Majority Strategy",
        ylabel="Minority Strategy",
        title="",
        yticks=0:15,
        ytickformat=x -> getindex.(Ref(strategy_names), Int.(x)),
        xticks=0:15,
        xtickformat=x -> getindex.(Ref(strategy_names), Int.(x)),
        aspect=1,
    )
    # Group by generation
    df = DataFrame(prevalence_matrix, string.(0:15))
    df.strategy_maj = 0:15
    df = stack(
        df, Not(:strategy_maj); variable_name=:strategy_min, value_name=:joint_prevalence
    )
    transform!(df, :strategy_min => ByRow(Fix1(parse, Int)) => :strategy_min)
    heatmap!(
        df.strategy_maj,
        df.strategy_min,
        df.joint_prevalence;
        colormap=cmap,
        colorrange=(0, 1),
    )
    Colorbar(fig[1, 2]; limits=(0, 1), colormap=cmap, vertical=true, label="Prevalence")
    for filetype in ("png", "pdf")
        save(
            "figures/abm/multi_pair_prevalence_heatmap_$(norm)_$(majority_benefit).$filetype",
            fig,
        )
        save("$norm.$filetype", fig)
    end
    return fig
end

pair_prevalence_heatmap(df_strategy_count; strategy_names)

selfjoin(x) = crossjoin(x, x; makeunique=true)

@chain begin

    # @pivot_wider(names_from =group, values_from = group_prevalence)
    # rename("true" => :maj_prevalence, "false" => :min_prevalence)

    # select(:update_step, :strategy, :strategy_1 => )
end

pair_prevalence(df_strategy_count)

combine(
    combine(
        groupby(pair_prevalence(df_strategy_count), :strategy_maj),
        :avg_joint_prevalence => sum,
    ),
    :avg_joint_prevalence_sum => sum,
)
