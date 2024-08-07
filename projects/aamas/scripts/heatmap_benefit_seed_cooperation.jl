using CSV
using DataFrames
using CairoMakie

benefits = 3:9
# coop_data = mapreduce(hcat, benefits) do benefit
#     df = CSV.read(
#         "projects/aamas/data/metrics/195_cooperation_$benefit.csv", DataFrame; stripwhitespace=true
#     )
#     df.cooperation_mean
# end

begin
    fig = Figure(; size=(600, 220))
    tick_labels = ["Always defect", "Group-agnostic", "Discriminatory"]
    ax = Axis(
        fig[1, 1];
        xlabel=rich(
            "Number of agents with Q-values initialised to strategy ",
            rich("\"Disc\""; font=:italic),
        ),
        ylabel="Benefit-to-cost ratio",
    )
    hmap = heatmap!(
        ax,
        0:2:50,
        benefits,
        (x, y) -> begin
            df = CSV.read(
                "projects/aamas/data/metrics/faster_195_cooperation_$(Int(y)).csv",
                DataFrame;
                stripwhitespace=true,
            )
            subset(df, :n_seeded_agents => ByRow(==(x))).cooperation_mean[1]
        end;
        colormap=:thermal,
        colorrange=(0, 1),
    )
    Colorbar(fig[:, 2], hmap; label="Cooperation level")#, tickformat = values -> ["$(round(Int, value*100))%" for value in values])
    for filetype in ("pdf", "png")
        save(
            "projects/aamas/figures/faster_heatmap_benefit_seed_cooperation.$filetype", fig
        )
    end
    fig
end
