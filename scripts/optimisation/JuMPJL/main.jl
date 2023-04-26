using IR
using StaticArrays
using Printf
using JuMP
using Ipopt

# Globals:
## Social rules
norm::Norm = iNorm(195)
red_strategy::Strategy  = iStrategy(12)
blue_strategy::Strategy = iStrategy(12)
benefit_sum_max::Int = 4 # Sum of each group's benefit
cost_sum_min::Int = 2 # Sum of each group's costs
cost_min = 1.0
benefit_max = 2.0
em_bound::Float64 = 0.1
pm_bound::Float64 = 0.1

# Solver options:
domain_eps::Float64 = 0.001

p = (;
    norm,
    red_strategy,
    blue_strategy,
    benefit_sum_max,
    benefit_max,
    cost_sum_min,
    cost_min,
    em_bound,
    pm_bound,
    domain_eps
)

# Functions that depend on globals (eek)
include("jump_integration.jl")

begin
    # Initialise model with Ipopt
    model = Model(Ipopt.Optimizer)

    # Register our custom objective function and auxiliary functions useful for constraints
    # register(model, :avg_payoffs, 15, (xs...) -> avg_payoffs(); autodiff=true)
    payoff_red = (args...) -> payoff_incumbent_i(args...; p=p, i_out=1)
    payoff_blue = (args...) -> payoff_incumbent_i(args...; p=p, i_out=2)
    register(model, :payoff_red, 15, payoff_red; autodiff=true)
    register(model, :payoff_blue, 15, payoff_blue; autodiff=true)

    # Add decision variables and their domains
    @variable(model, 0.5 <= pR <= 0.9) # Size of the majority

    @variable(model, domain_eps <= benefits[1:2] <= benefit_max) # Benefits
    @variable(model, cost_min <= costs[1:2] <= Inf) # Costs

    @variable(model, domain_eps <= judge_em <= em_bound) # Rates of execution mistakes of judge
    @variable(model, domain_eps <= red_em <= em_bound) # Rates of execution mistakes of red agents
    @variable(model, domain_eps <= blue_em <= em_bound) # Rates of execution mistakes of blue agents

    @variable(model, domain_eps <= judge_pm[1:3] <= pm_bound) # Rates of perception mistakes of judge
    @variable(model, domain_eps <= red_pm[1:2] <= pm_bound) # Rates of perception mistakes of agents
    @variable(model, domain_eps <= blue_pm[1:2] <= pm_bound) # Rates of perception mistakes of agents

    # @variable(model, domain_eps <= mistakes[1:10] <= 0.005, start = 0.01) # Rates of mistakes of judge and agents
    vars = (pR, benefits..., costs..., judge_em, judge_pm..., red_em, red_pm..., blue_em, blue_pm...,) # Put them all in a single variable

    # Add our non-linear objective function
    @NLobjective(model, Max, payoff_red(vars...) + payoff_blue(vars...)) # Maximise average payoffs...

    # ...subject to the constraint that the incumbent payoffs are larger or equal to
    # any possible mutant.
    println("Registering constraints 0 to 15")
    red_i = @evalpoly(2, red_strategy...)
    blue_i = @evalpoly(2, blue_strategy...)
    @show red_i blue_i
    for i in 0:15
        mutant_rule = Strategy(digits(i, base=2, pad=4))
        red_name = Symbol(string("payoff_red_mutant_", i))
        blue_name = Symbol(string("payoff_blue_mutant_", i))
        register(model, red_name, 15, (args...) -> mutant_payoff_i(args...; p, mutant_rule, out_i=1); autodiff=true)
        register(model, blue_name, 15, (args...) -> mutant_payoff_i(args...; p, mutant_rule, out_i=2); autodiff=true)
        if i != red_i
            @eval @NLconstraint(model, payoff_red(vars...) >= domain_eps + $(red_name)(vars...))
        end
        if i != blue_i
            @eval @NLconstraint(model, payoff_red(vars...) >= domain_eps + $(blue_name)(vars...))
        end
    end

    # Constraint on fairness
    @NLconstraint(model, payoff_red(vars...) >= 0.6 * payoff_blue(vars...))
    @NLconstraint(model, payoff_blue(vars...) >= 0.6 * payoff_red(vars...))

    # Constrain the utilities (although some are superfluous)
    begin
        # Benefits must be greater than costs
        @constraint(model, [i=1:2], costs[i] <= benefits[i])
        
        # @NLconstraint(model)
        # Sum of benefits can't exceed predetermined maximum
        @constraint(model, benefits[1] + benefits[2] <= benefit_sum_max)
        # Sum of costs must be greater than predetermined minimum
        @constraint(model, costs[1] + costs[2] >= cost_sum_min)
        # Groups must have the same benefits...
        @constraint(model, benefits[1] == benefits[2])
        # ...and costs
        @constraint(model, costs[1] == costs[2]) 
        # set_optimizer_attribute(model, "max_iter", 4000) # The solver can take a LOT of iterations before it finds an optimal solution
    end
end

unset_silent(model)
optimize!(model)

summarise_solution(model, vars; p)
both_payoffs(value.(vars)...; p)



# A few tests to see if things are as expected
# x0 = [0.9, 2.0, 2, 1, 1, 0.01, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00]
x0 = value.(vars)

# Do any mutants invade?
ESS_constraint(x0...; p)

payoff_red(x0...)
for i in 0:15
    println(mutant_payoff_i(x0...; p, mutant_rule=iStrategy(i), out_i=1))
end