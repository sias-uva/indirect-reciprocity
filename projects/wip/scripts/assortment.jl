using IR
using IRUtils
using StaticArrays
using LinearAlgebra
using OrdinaryDiffEq
using CairoMakie

function approx_eq_or_closer(ode_sol, stationary_point; kwargs...)
    if isapprox(ode_sol[end], stationary_point; kwargs...)
        return true
    else
        return sum(abs2, ode_sol[end - 1] - stationary_point) >
               sum(abs2, ode_sol[end] - stationary_point)
    end
end

# function step(U, p, t)
#     ρ, R̃, judges, agents = p
#     ΔU = map(CartesianIndices(U)) do ij
#         i, j = Tuple(ij)
#         # tot = 0.0
#         judge = judges[i]
#         donor = agents[j]
#         # Probability that group-j judged "good" by group-i given act as donor
#         # @show i, j
#         ℙ_j_good_i_pov = sum(1:size(U, 1)) do k
#             # Probability that j-donor i-judged good given k-recipient
#             # You're not meeting the average person from that group, you're
#             # meeting a specific person with a specific i-goodness and j-goodness
#             sameness = j == k
#             j_goodness = U[j, k]
#             i_goodness = U[i, k]
#             avg_judgement = calculate_reputation(
#                 i, j, donor, judge, sameness, j_goodness, i_goodness
#             )
#             # println("----")
#             # @show j k R[j, k] * π[k]
#             # tot += R[j, k] * π[k] * n_groups
#             R̃[j, k] * avg_judgement
#             # The above cannot be done in average, must be broken down otherwise
#             # norm never knows how to differentiate between good and bad actions.
#         end
#         # @show tot
#         ρ[j] * (ℙ_j_good_i_pov - U[ij])
#     end
#     # println("------------")
#     return ΔU
# end

let
    n_groups = 5
    U0 = 1 / 2 * ones(n_groups)
    π = 1:n_groups
    ρ = (1 / 2) * ones(n_groups)
    R = ones(n_groups, n_groups)
    R̃ = map(CartesianIndices((1:n_groups, 1:n_groups))) do ij
        i, j = Tuple(ij)
        R̃ij = R[ij] * π[j] / sum(R[i, k] * π[k] for k in 1:n_groups)
    end
    display(R̃)
    judge, agents = let
        ε̂ = 0.01
        α̂ = SA[0.00, 0.00, 0.00]
        ε = 0.01
        α = SA[0.00, 0.00]
        judge = Agent(iNorm(195), ε̂, α̂)
        agents = [Agent(iStrategy(12), ε, α) for _ in 1:n_groups]
        judge, agents
    end
    p = (ρ, R̃, judge, agents)
    tspan = (0.0, 100.0)
    prob = ODEProblem(step, U0, tspan, p)
    ode_sol = solve(prob, Tsit5())
    begin
        resolution = (300 * n_groups, 300 * n_groups)
        fig = Figure(; resolution)
        # axes = [Axis(fig[i, 1];) for i in 1:n_groups]
        for (i, j) in Iterators.product(1:n_groups, 1:n_groups)
            ax = Axis(fig[i, j];)
            # ylims!(ax, (0.5, 1))
            lines!(ax, ode_sol.t, t -> ode_sol(t)[i, j])
            # xlims!(ax, 90, 100)
        end
    end
    fig
end

let # maths
    n_groups = 4
    u0 = 1 / 2 * ones(n_groups)
    π = [0.6, 0.2, 0.1, 0.1]
    ρ = 1:n_groups
    R = ones(n_groups, n_groups)
    R̃ = map(CartesianIndices((1:n_groups, 1:n_groups))) do ij
        i, j = Tuple(ij)
        R̃ij = R[ij] * π[j] / sum(R[i, k] * π[k] for k in 1:n_groups)
    end
    judge, agents = let
        ε̂ = 0.01
        α̂ = SA[0.00, 0.00, 0.00]
        ε = 0.01, 0.1, 0.1, 0.1
        α = SA[0.00, 0.00]
        judge = Agent(iNorm(193), ε̂, α̂)
        agents = [Agent(iStrategy(12), ε[i], α) for i in 1:n_groups]
        judge, agents
    end
    J = map(CartesianIndices((1:n_groups, 1:n_groups))) do ij
        i, j = Tuple(ij)
        donor = agents[i]
        judge(i == j, 1, donor(i == j, 1)) - judge(i == j, 0, donor(i == j, 0))
    end
    A = (R̃ .* J) - I
    b = map(1:n_groups) do i
        donor = agents[i]
        sum(R̃[i, j] * judge(i == j, 0, donor(i == j, 0)) for j in 1:n_groups)
    end
    display(A)
    display(b)
    step(u, p, t) = A * u + b
    tspan = (0.0, 10.0)
    prob = ODEProblem(step, u0, tspan)
    ode_sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-8)
    true_solution = -A \ b
    approx_eq_or_closer(ode_sol, true_solution)
    begin
        resolution = (550, 500)
        fig = Figure(; resolution)
        ax = Axis(fig[1, 1]; xlabel="t", ylabel="Reputation")
        for i in 1:n_groups
            lines!(ax, ode_sol.t, t -> ode_sol(t)[i])
            hlines!(true_solution[i]; linestyle=:dash)
        end
        fig
    end
    display(fig)
end

let # maths
    n_groups = 2
    u0 = 1 / 2 * ones(n_groups)
    π = [0.8, 0.2]
    # ρ = SA[2, 1]
    R = ones(n_groups, n_groups)
    R̃ = map(CartesianIndices((1:n_groups, 1:n_groups))) do ij
        i, j = Tuple(ij)
        R̃ij = R[ij] * π[j] / sum(R[i, k] * π[k] for k in 1:n_groups)
    end
    judge, agents = let
        ε̂ = 0.01
        α̂ = SA[0.00, 0.00, 0.00]
        ε = 0.01, 0.1, 0.1, 0.1
        α = SA[0.00, 0.00]
        judge = Agent(iNorm(195), ε̂, α̂)
        agents = [Agent(iStrategy(12), ε[i], α) for i in 1:n_groups]
        judge, agents
    end
    J = map(CartesianIndices((1:n_groups, 1:n_groups))) do ij
        i, j = Tuple(ij)
        donor = agents[i]
        judge(i == j, 1, donor(i == j, 1)) - judge(i == j, 0, donor(i == j, 0))
    end
    A = SMatrix{2,2}((R̃ .* J) - I)
    b = SVector{2}(
        map(1:n_groups) do i
            donor = agents[i]
            sum(R̃[i, j] * judge(i == j, 0, donor(i == j, 0)) for j in 1:n_groups)
        end,
    )
    @show A b
    f(point) = A * point + b
    true_solution = -A \ b
    begin
        resolution = (500, 500)
        fig = Figure(; resolution)
        ax = Axis(fig[1, 1]; xlabel="Majority reputation", ylabel="Minority reputation")
        streamplot!(ax, f, 0 .. 1, 0 .. 1)
        scatter!(ax, true_solution; color=:black)
    end
    display(fig)
end
