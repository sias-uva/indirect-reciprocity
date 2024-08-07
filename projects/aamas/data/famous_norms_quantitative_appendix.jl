using IR
using StaticArrays
using DataFrames
using CairoMakie
using GeometryBasics
using ColorSchemes
using Format
using CSV
using Tidier

# CairoMakie.activate!()
using IRUtils

begin
    player_execution_mistake_rate = 0.01
    judge_execution_mistake_rate = 0.01
    player_perception_mistake_rate = SA[0.00, 0.00]
    judge_perception_mistake_rate = 0.00
    proportion_incumbents_majority = 0.9
    utilities = SA[10, 10, 1, 1]

    p = (;
        maj_em=player_execution_mistake_rate,
        min_em=player_execution_mistake_rate,
        judge_em=judge_execution_mistake_rate,
        maj_pm=player_perception_mistake_rate,
        min_pm=player_perception_mistake_rate,
        judge_pm=judge_perception_mistake_rate,
        prop_maj=proportion_incumbents_majority,
        utilities=utilities,
    )

    filters = [
    # [:majority_strat, :minority_strat] => ByRow((r, b) -> is_fair(r) && (b == 0 || !is_fair(b)))
]
end

df = generate_quadrant_df(find_ESS(p); p)
df_rl = CSV.read(
    "projects/aamas/data/rl_data_10_0:255.csv", DataFrame; stripwhitespace=true
)
df_joined = let
    df_egt = copy(df)
    transform!(
        df_egt,
        [:majority_strat, :minority_strat] .=>
            ByRow(categorise_strategy_old) .=>
                [:majority_strat_category, :minority_strat_category],
    )
    transform!(df_egt, :norm => ByRow(split_norm) => AsTable)
    subset!(df_egt, :ingroup_norm => ByRow(in((9, 8, 13, 12))))
    subset!(df_egt, :outgroup_norm => ByRow(in((9, 8, 13, 12))))
    gdf = groupby(df_egt, :norm)
    df_egt = combine(gdf) do sdf
        first(sort(sdf, [:is_ess, :cooperation, :majority_strat]; rev=true))
    end
    select!(
        df_egt,
        :norm,
        :ingroup_norm,
        :outgroup_norm,
        :majority_strat,
        :minority_strat,
        :cooperation,
        :fairness,
    )
    df_joined = @chain df_egt begin
        leftjoin(df_rl; on=:norm, makeunique=true, renamecols=("" => "_rl"))
        transform(:norm => ByRow(split_norm) => AsTable)
        subset(:ingroup_norm => ByRow(in((9, 8, 13, 12))))
        subset(:outgroup_norm => ByRow(in((9, 8, 13, 12))))
        transform(:fairness => ByRow(x -> isnan(x) ? 1.0 : x) => :fairness)
    end
    shortnorms = Dict(8 => "Sh", 9 => "SJ", 12 => "IS", 13 => "SS")
    shortstrats = Dict(0 => "AllD", 1 => "InvDisc", 2 => "Disc", 3 => "AllC")
    function split_strat(i)
        strat = iStrategy(i)
        outgroup_strat = evalpoly(2, reshape(strat[1, :], 2))
        ingroup_strat = evalpoly(2, reshape(strat[2, :], 2))
        return (; ingroup_strat, outgroup_strat)
    end
    sort!(df_joined, [:ingroup_norm, :outgroup_norm])
    select!(
        df_joined,
        [:ingroup_norm, :outgroup_norm] .=>
            ByRow(x -> shortnorms[x]) .=> [:ingroup_norm, :outgroup_norm],
        [:majority_strat, :minority_strat] .=>
            ByRow(split_strat) .=> [
                [:majority_outgroup_strat, :majority_ingroup_strat],
                [:minority_outgroup_strat, :minority_ingroup_strat],
            ],
        [:cooperation, :cooperation_rl, :fairness, :fairness_rl] .=>
            ByRow(x -> round(x; sigdigits=3)) .=>
                [:cooperation, :cooperation_rl, :fairness, :fairness_rl],
    )
    transform!(
        df_joined,
        [
            :majority_outgroup_strat,
            :majority_ingroup_strat,
            :minority_outgroup_strat,
            :minority_ingroup_strat,
        ] .=>
            ByRow(x -> shortstrats[x]) .=> [
                :majority_outgroup_strat,
                :majority_ingroup_strat,
                :minority_outgroup_strat,
                :minority_ingroup_strat,
            ],
    )
    select!(
        df_joined,
        [
            :ingroup_norm,
            :outgroup_norm,
            :majority_ingroup_strat,
            :majority_outgroup_strat,
            :minority_ingroup_strat,
            :minority_outgroup_strat,
            :cooperation,
            :cooperation_rl,
            :fairness,
            :fairness_rl,
        ],
    )
    rename(
        df_joined,
        [
            "In-norm",
            "Out-norm",
            "Mi-strat",
            "Mo-strat",
            "mi-strat",
            "mo-strat",
            "Cooperation (EGT)",
            "Cooperation (RL)",
            "Fairness (EGT)",
            "Fairness (RL)",
        ],
    )
end
CSV.write("projects/aamas/data/famous_metrics.csv", df_joined)

let
    begin # Data for plot
        df_egt = copy(df)
        transform!(
            df_egt,
            [:majority_strat, :minority_strat] .=>
                ByRow(categorise_strategy_old) .=>
                    [:majority_strat_category, :minority_strat_category],
        )
        # subset!(df_egt, :majority_strat => ByRow(!=(0)))
        # subset!(
        #     df_egt,
        #     # :norm => ByRow(in([150, 192, 195, 243])),
        #     # :majority_strat => ByRow(in([12, 9])),
        # )
        transform!(df_egt, :norm => ByRow(split_norm) => AsTable)
        subset!(df_egt, :ingroup_norm => ByRow(in((9, 8, 13, 12))))
        subset!(df_egt, :outgroup_norm => ByRow(in((9, 8, 13, 12))))
        gdf = groupby(df_egt, :norm)
        df_egt = combine(gdf) do sdf
            first(sort(sdf, [:is_ess, :cooperation, :majority_strat]; rev=true))
        end
        select!(
            df_egt,
            :norm,
            :ingroup_norm,
            :outgroup_norm,
            :majority_strat,
            :minority_strat,
            :cooperation,
            :fairness,
        )
        # transform!(df_egt, :norm => ByRow(split_norm) => AsTable)
    end
    df_egt

    ticks = 0:0.25:1
    fig = Figure(; resolution=(600, 600))
    ax = Axis(
        fig[1, 1];
        xlabel="Cooperativeness",
        ylabel="Fairness",
        # xticks=ticks,
        # yticks=ticks,
        title="",
        titlealign=:left,
        # aspect=DataAspect(),
    )
    offset = 0.025
    xlims = (0 - offset, 1 + offset)
    ylims = (0 - offset, 1 + offset)
    limits!(ax, xlims, ylims)

    df_new = @chain df_egt begin
        leftjoin(df_rl; on=:norm, makeunique=true, renamecols=("" => "_rl"))
        transform(:norm => ByRow(split_norm) => AsTable)
        subset(:ingroup_norm => ByRow(in((9, 8, 13, 12))))
        subset(:outgroup_norm => ByRow(in((9, 8, 13, 12))))
        transform(:fairness => ByRow(x -> isnan(x) ? 1.0 : x) => :fairness)
    end
    df_new
end
