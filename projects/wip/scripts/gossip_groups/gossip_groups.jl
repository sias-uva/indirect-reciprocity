using IR
using IRUtils
using StaticArrays
using LinearAlgebra
using OrdinaryDiffEq
# using SteadyStateDiffEq
using NonlinearSolve
using CairoMakie

# include("../misc.jl")

function approx_eq_or_closer(ode_sol, stationary_point; kwargs...)
    if isapprox(ode_sol[end], stationary_point; kwargs...)
        return true
    else
        return sum(abs2, ode_sol[end - 1] - stationary_point) >
               sum(abs2, ode_sol[end] - stationary_point)
    end
end

function calculate_reputation(i, j, donor, judge, sameness, j_goodness, i_goodness)
    jd(s, ig, jg) = judge(s, ig, donor(s, jg))
    return lerp(
        SA[
            # sameness = 0
            lerp(
                SA[
                    # donor sees agent as bad,
                    lerp(SA[ # judge sees agent as bad
                        jd(0, 0, 0),
                        # judge sees agent as good
                        jd(0, 1, 0),
                    ], i == j ? 0 : i_goodness)
                    # donor sees agent as good
                    lerp(SA[ # judge sees agent as bad
                        jd(0, 0, 1),
                        # judge sees agent as good
                        jd(0, 1, 1),
                    ], i == j ? 1 : i_goodness)
                ],
                j_goodness,
            ),
            # sameness = 1
            lerp(
                SA[
                    # donor sees agent as bad,
                    lerp(SA[jd(1, 0, 0), jd(1, 1, 0)], i == j ? 0 : i_goodness)
                    # donor sees agent as good
                    lerp(SA[jd(1, 0, 1), jd(1, 1, 1)], i == j ? 1 : i_goodness)
                ],
                j_goodness,
            ),
        ],
        sameness,
    )
end

function step(U, p, t)
    ρ, R̃, judges, agents = p
    ΔU = map(CartesianIndices(U)) do I
        i, j = Tuple(I)
        # tot = 0.0
        judge = judges[i]
        donor = agents[j]
        # Probability that group-j judged "good" by group-i given act as donor
        # @show i, j
        ℙ_j_good_i_pov = sum(1:size(U, 1)) do k
            # Probability that j-donor i-judged good given k-recipient
            # You're not meeting the average person from that group, you're
            # meeting a specific person with a specific i-goodness and j-goodness
            sameness = j == k
            j_goodness = U[j, k]
            i_goodness = U[i, k]
            avg_judgement = calculate_reputation(
                i, j, donor, judge, sameness, j_goodness, i_goodness
            )
            # println("----")
            # @show j k R[j, k] * π[k]
            # tot += R[j, k] * π[k] * n_groups
            R̃[j, k] * avg_judgement
            # The above cannot be done in average, must be broken down otherwise
            # norm never knows how to differentiate between good and bad actions.
        end
        # @show tot
        ρ[j] * (ℙ_j_good_i_pov - U[I])
    end
    # println("------------")
    return ΔU
end

let
    n_groups = 3
    U0 = rand(n_groups, n_groups)
    π = rand(n_groups)
    ρ = (1 / 2) * rand(n_groups)
    R = rand(n_groups, n_groups)
    R̃ = map(CartesianIndices((1:n_groups, 1:n_groups))) do I
        i, j = Tuple(I)
        R̃ij = R[I] * π[j] / sum(R[i, k] * π[k] for k in 1:n_groups)
    end
    # display(R̃)
    judges, agents = let
        ε̂ = 0.01
        α̂ = SA[0.00, 0.00, 0.00]
        ε = 0.01
        α = SA[0.00, 0.00]
        judges = [Agent(iNorm(195), ε̂, α̂) for _ in 1:n_groups]
        agents = [Agent(iStrategy(12), ε, α) for _ in 1:n_groups]
        judges, agents
    end
    p = (ρ, R̃, judges, agents)
    tspan = (0.0, 100.0)
    prob = ODEProblem(step, U0, tspan, p)
    @time ode_sol = solve(prob, Tsit5())
    @time ss_sol = solve(NonlinearProblem(prob), NewtonRaphson())
    @show approx_eq_or_closer(ode_sol, ss_sol.u)
    begin
        resolution = (300 * n_groups, 300 * n_groups)
        fig = Figure(; resolution)
        # axes = [Axis(fig[i, 1];) for i in 1:n_groups]
        for (i, j) in Iterators.product(1:n_groups, 1:n_groups)
            ax = Axis(fig[i, j];)
            # ylims!(ax, (0.5, 1))
            lines!(ax, ode_sol.t, t -> ode_sol(t)[i, j])
            hlines!(ss_sol.u[i, j]; linestyle=:dash, color=:black)
            # xlims!(ax, 90, 100)
        end
    end
    display(ss_sol)
    display(fig)
end

# let
#     n_groups = 3
#     U0 = 1 / 2 * ones(n_groups, n_groups)
#     π = 1:n_groups
#     ρ = (1 / 2) * ones(n_groups)
#     R = ones(n_groups, n_groups)
#     R̃ = map(CartesianIndices((1:n_groups, 1:n_groups))) do I
#         i, j = Tuple(I)
#         R̃ij = R[I] * π[j] / sum(R[i, k] * π[k] for k in 1:n_groups)
#     end
#     display(R̃)
#     judges, agents = let
#         ε̂ = 0.01
#         α̂ = SA[0.00, 0.00, 0.00]
#         ε = 0.01
#         α = SA[0.00, 0.00]
#         judges = [Agent(iNorm(193), ε̂, α̂) for _ in 1:n_groups]
#         agents = [Agent(iStrategy(12), ε, α) for _ in 1:n_groups]
#         judges, agents
#     end
#     p = (ρ, R̃, judges, agents)
#     # tspan = (0.0, 100.0)
#     # prob = ODEProblem(step, U0, tspan, p)
#     # ode_sol = solve(prob, Tsit5())
#     f(point) = step(point, p, 0.0)
#     # f(x, y) = step(SA[x, y], p, 0.0)
#     begin
#         resolution = (500, 500)
#         fig = Figure(; resolution)
#         ax = Axis(fig[1, 1])
#         arrows(0:0.1:1, 0:0.1:1, f)
#     end
#     display(fig)
# end
