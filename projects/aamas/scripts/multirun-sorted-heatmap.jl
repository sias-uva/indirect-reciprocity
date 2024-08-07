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

function generate_ticks_simple(values; nticks)
    return Int.(round.(range(0, length(values); length=2nticks)))[2:2:(end - 1)]
end

strategy_names = Dict{Int,String}(0 => "AllD", 3 => "pDisc", 12 => "Disc", 15 => "AllC")
for i in 0:15
    get!(strategy_names, i) do
        string(i)
    end
end

function generate_ticks(inputvalues; n_ticks=6, diff_min=3)
    lo, hi = extrema(inputvalues)
    value_step = (hi - lo) / n_ticks
    stepvalues = (lo + value_step / 2):value_step:(hi - value_step / 2)
    N = length(inputvalues)
    candidate_ticks = unique([findfirst(>=(v), inputvalues) - 1 for v in stepvalues])
    if length(candidate_ticks) > 1
        mat = map(Iterators.product(candidate_ticks, candidate_ticks)) do (x, y)
            y - x >= diff_min
        end
        if !reduce(|, mat)
            candidate_tick_indices = [[i] for (i, _) in enumerate(candidate_ticks)]
        else
            candidate_tick_indices = Vector{Int}[]
            for candidate_start in findall(reduce(|, mat; dims=2))
                candidate_tick_indices_list = Int[]
                next = candidate_start[1]
                while !isnothing(next)
                    push!(candidate_tick_indices_list, next)
                    next = findfirst(@views mat[next, :])
                end
                push!(candidate_tick_indices, candidate_tick_indices_list)
            end
        end
    else
        candidate_tick_indices = [[1]]
        # max_length_candidate_ticks = maximum(length, candidate_tick_indices, init=0)
        # filter!(list -> length(list) == max_length_candidate_ticks, candidate_tick_indices)
    end
    candidate_ticks_vec = map(candidate_tick_indices) do indices
        current_candidate_ticks = candidate_ticks[indices]
        n_ticks_to_generate = n_ticks - length(indices)
        tick_extrema = extrema(current_candidate_ticks)
        nogo_length = -reduce(-, tick_extrema) + 2diff_min
        other_ticks_range = clamp(N - nogo_length, 0, N)
        other_ticks_step = other_ticks_range / n_ticks_to_generate
        if other_ticks_range != 0
            other_ticks_unrounded = collect(
                (other_ticks_step / 2):other_ticks_step:(other_ticks_range - other_ticks_step / 2),
            )
            other_ticks_rounded =
                (Int ∘ Fix2(round, RoundNearestTiesUp)).(other_ticks_unrounded)
            for (i, val) in enumerate(other_ticks_rounded)
                if val in range((tick_extrema .+ (-diff_min, diff_min))...)
                    other_ticks_rounded[i] += nogo_length
                end
            end
            new_candidate_ticks = sort!(vcat(other_ticks_rounded, current_candidate_ticks))
        else
            new_candidate_ticks = candidate_ticks
        end
        filter!(in(1:N), new_candidate_ticks)
        new_candidate_ticks
    end
    max_length_ticks = maximum(length, candidate_ticks_vec; init=0)
    filter!(vec -> length(vec) == max_length_ticks, candidate_ticks_vec)
    scores = map(candidate_ticks_vec) do candidate_ticks
        # minimise the variance of the distances between ticks
        # optimal is all same distance
        var(diff([0, candidate_ticks..., N]); corrected=false)
    end
    return candidate_ticks_vec[argmin(scores)]
end

begin
    # Magic constants
    strategy_range = 0:15

    # Global simulation variables
    Z = population_size = 50
    majority_proportion = 0.9
    generations = population_size * 10
    generation_length = 10 * population_size
    n_training_interactions = generations * generation_length
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
    majority_benefit = 7
    minority_benefit = 7
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

# norm = 150
for norm in (0, 150, 192, 195, 243)# (155:255)
    @show norm
    begin
        granular_output = DataFrame(
            :seed => Int[],
            :cooperation => Float64[],
            :fairness => Float64[],
            :minority_prevalence => Matrix{Float64}[],
            :majority_prevalence => Matrix{Float64}[],
        )
        n_runs = 10
        seeds = 1:n_runs
        coop_fairness_vec = zeros(SVector{2,Float64}, n_runs)
        for (i_seed, seed) in enumerate(seeds)
            @show i_seed
            rng = Xoshiro(seed)
            abm = initialise_rlabm(
                norm,
                judge_characteristics,
                agent_characteristics,
                utilities,
                global_simulation_variables;
                rng=rng,
            )
            data = (;
                strategies=(;
                    minority_strategies=Vector{Int64}[], majority_strategies=Vector{Int64}[]
                ),
                df_cooperation=DataFrame(
                    :donor => Int64[], :recipient => Int64[], :action => Bool[]
                ),
            )
            train_and_collect!(abm, data)
            coop_fairness = get_coop_fairness(abm, data)
            coop_fairness_vec[i_seed] = coop_fairness
            prevalence = strategy_prevalence(abm, data)
            push!(granular_output, (seed, coop_fairness..., prevalence...))
        end
        granular_output
    end

    function strategy_prevalence_multirun_heatmap(granular_output; strategy_names)
        fig = Figure(; resolution=(600, 500))
        cmap = :viridis
        ax_maj = Axis(
            fig[1, 1];
            ylabel="Run",
            xlabel="Strategy",
            title="Majority",
            # xticks=0:15,
            xtickformat=x -> getindex.(Ref(strategy_names), Int.(x)),
            # xaxisposition=:top,
            # yticks=1:n_runs,
            # ytickformat=x -> getindex.(Ref(sort(granular_output.cooperation, rev=true)), Int.(x)),
            # aspect=1,
        )
        ax_min = Axis(
            fig[1, 2];
            ylabel="Run",
            xlabel="Strategy",
            title="Minority",
            # xticks=0:15,
            xtickformat=x -> getindex.(Ref(strategy_names), Int.(x)),
            # aspect=1,
            # xticklabelrotation=0,
        )

        # linkaxes!(ax_maj, ax_min)
        # linkyaxes!(ax_maj, ax_coop)
        hideydecorations!(ax_min; grid=false)
        # Group by generation
        df = sort(granular_output, order(:cooperation))
        df.order = 1:nrow(df)
        df_out = DataFrame(
            "order" => Int[], "group" => Bool[], (string.(0:15) .=> Ref(Float64[]))...
        )
        for row in eachrow(df)
            mip, map = row[[:minority_prevalence, :majority_prevalence]]
            for (group, mat) in enumerate((mip, map))
                new_row = (row.order, Bool(group - 1), mean.(eachcol(mat))...)
                push!(df_out, new_row)
            end
        end
        df_long = stack(df_out, 3:18; variable_name=:strategy, value_name=:prevalence)
        df_long = stack(df_out, 3:18; variable_name=:strategy, value_name=:prevalence)
        df_xticks = @chain df_long begin
            groupby([:strategy, :group])
            combine(:prevalence => maximum => :pm)
            subset(:pm => ByRow(>(0.2)))
        end

        ax_maj.xticks = parse.(Int, subset(df_xticks, :group).strategy)
        ax_min.xticks = parse.(Int, subset(df_xticks, :group => .!).strategy)
        for (group, ax) in zip((false, true), (ax_min, ax_maj))
            df_sub = subset(df_long, :group => ByRow(==(group)))
            # df.run = 1:nrow(df)
            @. df_sub.strategy = parse.(Int, df_sub.strategy)
            # @show df.strategy
            heatmap!(
                ax,
                df_sub.strategy,
                df_sub.order,
                df_sub.prevalence;
                colormap=cmap,
                colorrange=(0, 1),
            )
        end
        # split coop values into quintiles
        n_ticks = 8
        tick_diff_min = 4 # Ticks must be at least X apart
        # yticks, ytickformat_vec = generate_ticks(df.cooperation)
        # yticks = generate_ticks(df.cooperation; diff_min=tick_diff_min)
        yticks = generate_ticks_simple(df.cooperation; nticks=n_ticks)

        ax_coop = Axis(
            fig[1, 3];
            ylabel="Average cooperation in run",
            yaxisposition=:right,
            yticks,
            ytickformat=ys -> ["$(round.(df.cooperation[Int(y)]; digits=3))" for y in ys],
        )
        # hideydecorations!(ax_coop; label=false)
        hidexdecorations!(ax_coop; grid=false)
        heatmap!(
            ax_coop,
            ones(size(df.order)),
            df.order,
            df.cooperation;
            colormap=:thermal,
            colorrange=(0, 1),
        )
        for ax in (ax_maj, ax_min, ax_coop)
            hlines!(ax, yticks; color=:grey, linestyle=:dash)
        end
        # Strategies to highlight
        theoretical_df = generate_quadrant_df(find_ESS(p); p)
        theoretical_df = @chain theoretical_df begin
            subset(
                :norm => ByRow(==(norm)),
                [:majority_strat, :minority_strat] => ByRow((x, y) -> !(x == y == 0)),
            )
        end
        cb = Colorbar(
            fig[2, :];
            limits=(0, 1),
            colormap=cmap,
            vertical=false,
            label="Strategy prevalence within population",
            flipaxis=false,
        )
        cb.tellwidth = true
        colsize!(fig.layout, 3, Auto(0.1))
        resize_to_layout!(fig)
        for filetype in ("png", "pdf")
            save(
                "projects/aamas/figures/prevalence-heatmap/multirun_prevalence_heatmap_$(norm)_$(update_reputation_probability).$filetype",
                fig,
            )
        end
        # highlighed_majority = theoretical_df.majority_strat
        # highlighted_minority = theoretical_df.minority_strat
        # for (axis, highlighted_strategies) in zip((ax_maj, ax_min), (highlighed_majority, highlighted_minority))

        #     vline_cmap = cgrad(:buda, Int(clamp(length(highlighed_majority), 2, Inf)); rev=true, categorical=true)

        #     for (i, strat) in enumerate(highlighted_strategies)
        #         vlines!(axis, [strat - 0.5, strat + 0.5], color=vline_cmap[i], label="", linewidth=1.2)
        #     end
        # end
        # for filetype in ("png", "pdf")
        #     save(
        #         "projects/aamas/figures/multirun_prevalence_heatmap_$(norm)_$(update_reputation_probability)_with_lines.$filetype",
        #         fig,
        #     )
        # end
        display(fig)
        return fig
    end
    strategy_prevalence_multirun_heatmap(granular_output; strategy_names)
end

### TODO: move to another file? a notebook?

function get_granular_data(
    norm,
    judge_characteristics,
    agent_characteristics,
    utilities,
    global_simulation_variables;
    n_runs=5,
    seeds=1:n_runs,
)
    granular_output = DataFrame(
        :seed => Int[],
        :cooperation => Float64[],
        :fairness => Float64[],
        :minority_prevalence => Matrix{Float64}[],
        :majority_prevalence => Matrix{Float64}[],
    )
    # Magic constants
    strategy_range = 0:15

    coop_fairness_vec = zeros(SVector{2,Float64}, n_runs)
    for (i_seed, seed) in enumerate(seeds)
        @show i_seed
        rng = Xoshiro(seed)
        abm = initialise_rlabm(
            norm,
            judge_characteristics,
            agent_characteristics,
            utilities,
            global_simulation_variables;
            rng=rng,
        )
        data = (;
            strategies=(;
                minority_strategies=Vector{Int64}[], majority_strategies=Vector{Int64}[]
            ),
            df_cooperation=DataFrame(
                :donor => Int64[], :recipient => Int64[], :action => Bool[]
            ),
        )
        train_and_collect!(abm, data)
        coop_fairness = get_coop_fairness(abm, data)
        coop_fairness_vec[i_seed] = coop_fairness
        prevalence = strategy_prevalence(abm, data)
        push!(granular_output, (seed, coop_fairness..., prevalence...))
    end
    return granular_output
end

begin # Multiple plots in one figure
    norms = (195, 211, 243)
    granular_output_norms = mapreduce(vcat, norms) do norm
        go = get_granular_data(
            norm,
            judge_characteristics,
            agent_characteristics,
            utilities,
            global_simulation_variables;
            n_runs=50,
        )
        go.norm .= norm
        go
    end
end

let
    let
        fig = Figure(; resolution=(600, 1100))
        cmap = :viridis
        gtop = fig[1, 1] = GridLayout()
        gmid = fig[2, 1] = GridLayout()
        gbot = fig[3, 1] = GridLayout()
        normnames = Dict(
            [192, 195, 211, 243] .=>
                ["Shunning", "SternJudging", "Norm–211", "SimpleStanding"],
        )

        # Group by generation
        for (sdf, grid) in zip(groupby(granular_output_norms, :norm), (gtop, gmid, gbot))
            sdf_norm = only(unique(sdf.norm))
            ax_maj = Axis(
                grid[1, 1];
                ylabel="Run",
                xlabel="Strategy",
                title="Majority",
                xtickformat=x -> getindex.(Ref(strategy_names), Int.(x)),
                xticklabelrotation=π / 2,
            )
            ax_min = Axis(
                grid[1, 2];
                ylabel="Run",
                xlabel="Strategy",
                title="Minority",
                xtickformat=x -> getindex.(Ref(strategy_names), Int.(x)),
                xticklabelrotation=π / 2,
            )
            hideydecorations!(ax_min; grid=false)
            df = sort(sdf, order(:cooperation))
            df.order = 1:nrow(df)
            df_out = DataFrame(
                "order" => Int[], "group" => Bool[], (string.(0:15) .=> Ref(Float64[]))...
            )
            for row in eachrow(df)
                mip, map = row[[:minority_prevalence, :majority_prevalence]]
                for (group, mat) in enumerate((mip, map))
                    new_row = (row.order, Bool(group - 1), mean.(eachcol(mat))...)
                    push!(df_out, new_row)
                end
            end
            df_long = stack(df_out, 3:18; variable_name=:strategy, value_name=:prevalence)
            df_xticks = @chain df_long begin
                groupby([:strategy, :group])
                combine(:prevalence => maximum => :pm)
                subset(:pm => ByRow(>(0.25)))
            end

            ax_maj.xticks = parse.(Int, subset(df_xticks, :group).strategy)
            ax_min.xticks = parse.(Int, subset(df_xticks, :group => .!).strategy)
            for (group, ax) in zip((false, true), (ax_min, ax_maj))
                df_sub = subset(df_long, :group => ByRow(==(group)))
                # df.run = 1:nrow(df)
                @. df_sub.strategy = parse.(Int, df_sub.strategy)
                # @show df.strategy
                heatmap!(
                    ax,
                    df_sub.strategy,
                    df_sub.order,
                    df_sub.prevalence;
                    colormap=cmap,
                    colorrange=(0, 1),
                )
            end
            # split coop values into quintiles
            n_ticks = 5
            tick_diff_min = 4 # Ticks must be at least X apart
            # yticks, ytickformat_vec = generate_ticks(df.cooperation)
            # yticks = generate_ticks(df.cooperation; diff_min=tick_diff_min)
            yticks = generate_ticks_simple(df.cooperation; nticks=n_ticks)

            ax_coop = Axis(
                grid[1, 3];
                ylabel="Average cooperation in run",
                yaxisposition=:right,
                yticks,
                ytickformat=ys ->
                    ["$(round.(df.cooperation[Int(y)]; digits=3))" for y in ys],
            )
            hidexdecorations!(ax_coop; grid=false)
            heatmap!(
                ax_coop,
                ones(size(df.order)),
                df.order,
                df.cooperation;
                colormap=:thermal,
                colorrange=(0, 1),
            )
            for ax in (ax_maj, ax_min, ax_coop)
                hlines!(ax, yticks; color=:grey, linestyle=:dash)
            end
            Label(
                grid[0, :],
                "Prevalence of Strategies under $(normnames[sdf_norm])";
                word_wrap=true,
                font=:bold,
            )
        end
        n_plots = length(norms)
        cb = Colorbar(
            fig[n_plots + 1, :];
            limits=(0, 1),
            colormap=cmap,
            vertical=false,
            label="Strategy prevalence within population",
            flipaxis=false,
        )
        cb.tellwidth = true

        colsize!(gtop, 3, Auto(0.07))
        colsize!(gmid, 3, Auto(0.07))
        colsize!(gbot, 3, Auto(0.07))
        resize_to_layout!(fig)
        for filetype in ("png", "pdf")
            save(
                "projects/aamas/figures/prevalence-heatmap/prevalence_heatmap_norms_$norms.$filetype",
                fig,
            )
        end
        display(fig)
    end

    # # Strategies to highlight
    # theoretical_df = generate_quadrant_df(find_ESS(p); p)
    # theoretical_df = @chain theoretical_df begin
    #     subset(
    #         :norm => ByRow(==(norm)),
    #         [:majority_strat, :minority_strat] => ByRow((x, y) -> !(x == y == 0)),
    #     )
    # end
end
