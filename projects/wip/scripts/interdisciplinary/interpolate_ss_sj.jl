using IR
using StaticArrays
using DataFrames
using CairoMakie
using GeometryBasics
using ColorSchemes
using Format

# CairoMakie.activate!()

include("../norms.jl") # norms, simple_norms|
include("../misc.jl")

filters = [:norm => ByRow(in(norms_being_analysed))]
norms_being_analysed = [195, 211, 227, 243]

begin
    # ideal parameters
    player_execution_mistake_rate = 0.01
    judge_execution_mistake_rate = 0.01
    player_perception_mistake_rate = 0.0
    judge_perception_mistake_rate = 0.0
    proportion_incumbents_majority = 0.8
    utilities = SA[2, 2, 1, 1]

    perturbed_low = 0.057
    perturbed_high = 0.02

    p_ideal = (;
        maj_em=player_execution_mistake_rate,
        min_em=player_execution_mistake_rate,
        judge_em=judge_execution_mistake_rate,
        maj_pm=player_perception_mistake_rate,
        min_pm=player_perception_mistake_rate,
        judge_pm=judge_perception_mistake_rate,
        prop_maj=proportion_incumbents_majority,
        utilities=utilities,
    )

    p_fair_perturbation = (;
        maj_em=perturbed_low,
        min_em=perturbed_low,
        judge_em=perturbed_low,
        maj_pm=0.00,
        min_pm=0.00,
        judge_pm=SA[0.00, perturbed_high, 0.00],
        prop_maj=proportion_incumbents_majority,
        utilities=utilities,
    )

    p_minority_perturb = (;
        maj_em=0.01,
        min_em=perturbed_low,
        judge_em=0.01,
        maj_pm=0.00,
        min_pm=0.00,
        judge_pm=SA[0.00, perturbed_high, 0.00],
        prop_maj=proportion_incumbents_majority,
        utilities=utilities,
    )

    p_majority_perturb = (;
        maj_em=perturbed_low,
        min_em=0.01,
        judge_em=0.01,
        maj_pm=0.00,
        min_pm=0.00,
        judge_pm=SA[0.00, perturbed_high, 0.00],
        prop_maj=proportion_incumbents_majority,
        utilities=utilities,
    )

    list_of_properties = [
        p_ideal, p_fair_perturbation, p_minority_perturb, p_majority_perturb
    ]

    data = map(list_of_properties) do p
        df = find_ESS(p)
        subset!(
            df,
            :norm => ByRow(in(norms_being_analysed)),
            [:majority_strat, :minority_strat] .=> ByRow(==(12)),
            # [:majority_strat, :minority_strat] => ByRow((x, y) -> !(x == y == 0))
        )
        quadrant_df = generate_quadrant_df(df; p)
        select!(quadrant_df, Not(:is_ess), Not(:quadrant))
        quadrant_df
    end

    bit_meanings = Dict(1 => "Bad-Out", 2 => "Bad-In", 3 => "Good-Out", 4 => "Good-In")

    for (i, (df, p)) in enumerate(zip(data, list_of_properties))
        # @show i
        for norm in norms_being_analysed
            if isempty(subset(df, :norm => ByRow(==(norm))))
                judge, majority, minority = get_agents(norm, 12, 12; p)
                invaded_group, _, invading_strat, incumbent_payoff, mutant_payoff = invader(
                    judge, majority, minority, p.prop_maj, p.utilities
                )
                bit_differences = invading_strat .!= iStrategy(12)
                println("Group: $invaded_group")
                println("Strategy $(evalpoly(2, vec(invading_strat)))")
                for bit in findall(vec(bit_differences))
                    println(
                        "Bit $bit: $(bit_meanings[bit]), $(Int(iStrategy(12)[bit])) → $(Int(invading_strat[bit]))",
                    )
                end
                # println("$incumbent_payoff < $mutant_payoff")
                println("")
            end
        end
    end

    let
        markersize = 15
        data_name = ("No one", "Both groups", "Minority", "Majority")
        fig = Figure(; resolution=(650, 550))
        ax = Axis(
            fig[1, 1];
            aspect=DataAspect(),
            xlabel="Cooperativeness of Society",
            ylabel="Fairness of Society",
        )
        cmap = cgrad(:Hiroshige, 4; rev=true, categorical=true)
        markers = (:circle, :utriangle, :star5, :cross)
        for (i, d) in enumerate(data)
            foreach(enumerate(eachrow(d))) do (j, row)
                scatter!(
                    ax,
                    row.cooperation,
                    row.fairness;
                    markersize,
                    color=j,
                    marker=markers[i],
                    colormap=cmap,
                    strokewidth=1,
                    label=data_name[i],
                    colorrange=(1, 4),
                )
            end
        end
        # Make custom legend:
        shapes = [
            MarkerElement(; color=:black, marker, markersize, strokewidth=1) for
            marker in markers
        ]
        color_elements = [
            MarkerElement(; color, marker=:circle, markersize, strokewidth=1) for
            color in getindex.(Ref(cmap), 1:4)
        ]
        shape_labels = [data_name...]
        these_norm_names = Dict(195 => "SJ", 211 => "SJ/SS", 227 => "SS/SJ", 243 => "SS")
        color_labels = getindex.(Ref(these_norm_names), norms_being_analysed)
        Legend(
            fig[2, 1],
            [shapes, color_elements],
            [shape_labels, color_labels],
            ["Who was perturbed?", "Which norm? (In/Out)"];
            nbanks=2,
            orientation=:horizontal,
            tellwidth=false,
            tellheight=true,
        )
        offset = 0.01
        lims = (0.8 - offset, 1 + offset)
        limits!(ax, lims, lims)
        fig
    end
end

let
    1
end
1
1
1
# function is_interpolated_norm_ess(bad_out_coop, bad_in_coop; p)
#     # Setup 
#     majority_strat = iStrategy(12)
#     minority_strat = iStrategy(12)
#     norm = reshape(SA[1, 1, 0, 0, bad_out_coop, bad_in_coop, 1, 1], Size(2, 2, 2))
#     judge = Agent(norm, p.judge_em, p.judge_pm)
#     majority = Agent(majority_strat, p.maj_em, p.maj_pm)
#     minority = Agent(minority_strat, p.min_em, p.min_pm)

#     # Calculate payoffs
#     majority_rep, minority_rep = stationary_incumbent_reputations(judge, majority, minority, p.prop_maj)
#     majority_payoff, minority_payoff = incumbent_payoffs(
#         majority, minority, majority_rep, minority_rep, p.prop_maj, p.utilities
#     )
#     # @show majority_rep minority_rep

#     # Calculate fairness and cooperation
#     prd = p_donates(majority, majority_rep, minority_rep, p.prop_maj)
#     pbd = p_donates(minority, minority_rep, majority_rep, 1 - p.prop_maj)
#     fairness = let
#         lower, higher = minmax(majority_payoff, minority_payoff)
#         lower/higher
#     end
#     cooperation = p.prop_maj * prd + (1 - p.prop_maj) * pbd
#     return cooperation, fairness, is_ESS(judge, majority, minority, p.prop_maj, p.utilities)
# end

# function who_invades_norm(bad_out_coop, bad_in_coop; p)
#     majority_strat = iStrategy(12)
#     minority_strat = iStrategy(12)
#     norm = reshape(SA[1, 1, 0, 0, bad_out_coop, bad_in_coop, 1, 1], Size(2, 2, 2))
#     judge = Agent(norm, p.judge_em, p.judge_pm)
#     majority = Agent(majority_strat, p.maj_em, p.maj_pm)
#     minority = Agent(minority_strat, p.min_em, p.min_pm)
#     inv = invader(judge, majority, minority, p.prop_maj, p.utilities)
#     return inv === nothing ? 16 : evalpoly(2, reshape(inv[3], Size(4)))
# end

xys = 0:0.01:1
all(is_interpolated_norm_ess(boc, bic; p)[3] for (boc, bic) in Iterators.product(xys, xys))
