using IR
using StaticArrays
using LinearAlgebra
using OrdinaryDiffEq
using NonlinearSolve
using CairoMakie

include("misc.jl")

function approx_eq_or_closer(ode_sol, stationary_point)
    if ode_sol[end] ≈ stationary_point
        return true
    else
        return sum(abs2, ode_sol[end - 1] - stationary_point) >
               sum(abs2, ode_sol[end] - stationary_point)
    end
end

function main()
    player_execution_mistake_rate = 0.01
    judge_execution_mistake_rate = 0.01
    player_perception_mistake_rate = SA[0.00, 0.00]
    judge_perception_mistake_rate = 0.00
    proportion_incumbents_majority = 0.8

    p = (;
        maj_em=player_execution_mistake_rate,
        min_em=player_execution_mistake_rate,
        judge_em=judge_execution_mistake_rate,
        maj_pm=player_perception_mistake_rate,
        min_pm=player_perception_mistake_rate,
        judge_pm=judge_perception_mistake_rate,
        prop_maj=proportion_incumbents_majority,
    )

    judge, majority, minority = get_agents(192, 12, 12; p)
    prop_maj = p.prop_maj # Rename for brevity
    JR(x, y) = judge(x, y, majority(x, y))
    JB(x, y) = judge(x, y, minority(x, y))
    A = SA[
        prop_maj*(JR(1, 1) - JR(1, 0)) (1 - prop_maj)*(JR(0, 1) - JR(0, 0))
        prop_maj*(JB(0, 1) - JB(0, 0)) (1 - prop_maj)*(JB(1, 1) - JB(1, 0))
    ]
    b = SA[lerp(SA[JR(0, 0), JR(1, 0)], prop_maj), lerp(SA[JB(1, 0), JB(0, 0)], prop_maj)]
    Ã = A - I
    display(Ã)
    f(u, p, t) = Ã * u + b
    u0 = SA[1 / 2, 1 / 2]
    tspan = (0.0, 10.0)
    prob = ODEProblem(f, u0, tspan)
    ode_sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-8)
    true_solution = stationary_incumbent_reputations(judge, majority, minority, p.prop_maj)
    ss_sol = solve(NonlinearProblem(prob), NewtonRaphson())
    approx_eq_or_closer(ode_sol, true_solution)
    display(true_solution)
    display(ss_sol)
    # b_analytical = @benchmark stationary_incumbent_reputations($judge, $majority, $minority, $p.prop_maj)
    # display(b_analytical)
    # b_sciml = @benchmark solve(NonlinearProblem($prob), NewtonRaphson())
    # display(b_sciml)
    begin
        resolution = (550, 500)
        fig = Figure(; resolution)
        ax = Axis(fig[1, 1];)
        lines!(ode_sol.t, first.(ode_sol.u))
        lines!(ode_sol.t, last.(ode_sol.u))
        hlines!.(true_solution, linestyle=:dash)
        fig
    end
end

main()
