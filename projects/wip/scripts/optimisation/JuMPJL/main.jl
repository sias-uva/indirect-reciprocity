using IR
using IR: Norm, Strategy
using StaticArrays

using JuMP
import MathOptInterface as MOI
using Ipopt

using Printf
using Plots

include("jump_integration.jl")

# Globals:
## Social rules
norm::Norm = iNorm(195)
majority_strategy::Strategy = iStrategy(12)
minority_strategy::Strategy = iStrategy(12)
benefit_sum_max::Int = 4 # Sum of each group's benefit
cost_sum_min::Int = 2 # Sum of each group's costs
cost_min = 1.0
benefit_max = 8.0
em_bound::Float64 = 0.01
pm_bound::Float64 = 0.01

# Solver options:
domain_eps::Float64 = 0.00001

p = (;
    norm,
    majority_strategy,
    minority_strategy,
    benefit_sum_max,
    benefit_max,
    cost_sum_min,
    cost_min,
    em_bound,
    pm_bound,
    domain_eps,
)

function frontier(p)
    coop_fairness = Tuple{Float64,Float64}[]
    for fairness_bound in 0.1:0.1:1.0
        @show fairness_bound
        model = create_model(fairness_bound, p)
        vars = all_variables(model)
        cons = all_constraints(model; include_variable_in_set_constraints=false)
        nl_cons = all_nonlinear_constraints(model)

        # @show cons[37]
        set_silent(model)
        optimize!(model)

        prop_majority, utilities, judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm = unpack_x(
            vars
        )
        # summarise_solution(model, vars; p)
        # @show objective_value(model)
        # @show termination_status(model)
        # @show termination_status(model) == LOCALLY_SOLVED
        @show ESS_constraint(value.(vars)...; p)
        ratio_lmao =
            payoff_incumbent_i(value.(vars)...; p, i_out=1) /
            payoff_incumbent_i(value.(vars)...; p, i_out=2)
        ip = let
            prop_majority, utilities, judge_em, judge_pm, maj_em, maj_pm, min_em, min_pm = unpack_x(
                vars
            )
            judge = Agent(p.norm, value(judge_em), value.(judge_pm))
            majority = Agent(p.majority_strategy, value(maj_em), value.(maj_pm))
            minority = Agent(p.minority_strategy, value(min_em), value.(min_pm))
            R★, B★ = stationary_incumbent_reputations(
                judge, majority, minority, value(prop_majority)
            )
            ip = incumbent_payoffs(
                majority, minority, R★, B★, value(prop_majority), value.(utilities)
            )
        end

        @show termination_status(model)
        if termination_status(model) != LOCALLY_SOLVED
            push!(coop_fairness, (0.0, fairness_bound))
        else
            @show ip ratio_lmao
            push!(coop_fairness, cooperation_fairness(vars))
        end
    end
    return coop_fairness
end

cf = frontier(p)
cooperation_levels = first.(cf)
fairness_levels = last.(cf)

# unset_silent(model)
# optimize!(model)

# A few tests to see if things are as expected
# x0 = [0.9, 2.0, 2, 1, 1, 0.01, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00]
x0 = value.(vars)

# Do any mutants invade?
ESS_constraint(x0...; p)

payoff_majority(x0...)
for i in 0:15
    println(mutant_payoff_i(x0...; p, mutant_rule=iStrategy(i), out_i=1))
end
