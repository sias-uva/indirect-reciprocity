using IR
using StaticArrays
using Printf
using JuMP
using Ipopt

include("jump_integration.jl")

# Globals:
## Social rules
norm::Norm = iNorm(247)
red_strategy::Strategy  = iStrategy(8)
blue_strategy::Strategy = iStrategy(0)
benefit_sum_max::Int = 4 # Sum of each group's benefit
cost_sum_min::Int = 2 # Sum of each group's costs
em_bound::Float64 = 0.5
pm_bound::Float64 = 0.5

# Solver options:
domain_eps::Float64 = 0.0001

function build_model(norm, red_strategy, blue_strategy, benefit_sum_max, cost_sum_min, em_bound, pm_bound, domain_eps)
    # Initialise model with Ipopt
    model = Model(Ipopt.Optimizer)

    # Register our custom objective function and auxiliary functions useful for constraints
    register(model, :avg_payoffs, 15, avg_payoeffs; autodiff=true)
    register(model, :payoff_red, 15, payoff_red; autodiff=true)
    register(model, :payoff_blue, 15, payoff_blue; autodiff=true)

    # Add decision variables and their domains
    @variable(model, 0.5 <= pR <= 0.9) # Size of the majority

    @variable(model, domain_eps <= utilities[1:4] <= Inf) # Benefits and costs

    @variable(model, domain_eps <= judge_em <= em_bound) # Rates of execution mistakes of judge
    @variable(model, domain_eps <= red_em <= em_bound) # Rates of execution mistakes of red agents
    @variable(model, domain_eps <= blue_em <= em_bound) # Rates of execution mistakes of blue agents

    @variable(model, domain_eps <= judge_pm[1:3] <= pm_bound) # Rates of perception mistakes of judge
    @variable(model, domain_eps <= red_pm[1:2] <= pm_bound) # Rates of perception mistakes of agents
    @variable(model, domain_eps <= blue_pm[1:2] <= pm_bound) # Rates of perception mistakes of agents

    # @variable(model, domain_eps <= mistakes[1:10] <= 0.005, start = 0.01) # Rates of mistakes of judge and agents
    vars = (pR, utilities..., judge_em, judge_pm..., red_em, red_pm..., blue_em, blue_pm...,) # Put them all in a single variable

    # Add our non-linear objective function
    @NLobjective(model, Max, avg_payoffs(vars...)) # Maximise average payoffs...

    # ...subject to the constraint that the incumbent payoffs are larger or equal to
    # any possible mutant.
    println("Registering constraints 0 to 15")
    red_i = @evalpoly(2, red_strategy...)
    blue_i = @evalpoly(2, blue_strategy...)
    @show red_i blue_i
    for i in 0:15
        print(i)
        # We do a really nasty metaprogramming trick to make this work. Please don't do this ever.
        red_name = Symbol(string("payoff_red_mutant_", i))
        blue_name = Symbol(string("payoff_blue_mutant_", i))
        @eval begin
            if $red_i != $i 
                register(model, Symbol(string("payoff_red_mutant_", $i)), 15, $(red_name); autodiff=true)
                @NLconstraint(model, payoff_red(vars...) >= domain_eps + $(red_name)(vars...))
                print(".")
            else
                print("r")
            end
            if $blue_i != $i
                register(model, Symbol(string("payoff_blue_mutant_", $i)), 15, $(blue_name); autodiff=true)
                @NLconstraint(model, payoff_blue(vars...) >= domain_eps + $(blue_name)(vars...))
                print(".")
            else
                print("b")
            end
        end
        mod(i, 4)==3 && println()
    end

    
    # @NLconstraint(model, payoff_red(vars...) >= 0.6 * payoff_blue(vars...))
    # @NLconstraint(model, payoff_blue(vars...) >= 0.6 * payoff_red(vars...))

    # Constrain the utilities (although some are superfluous)
    begin
        @NLconstraint(model, 0 <= utilities[1] + utilities[2] <= benefit_sum_max)
        @NLconstraint(model, utilities[3] + utilities[4] >= cost_sum_min)
        @NLconstraint(model, utilities[1] == utilities[2]) # Groups must have the same benefits...
        @NLconstraint(model, utilities[3] == utilities[4]) # ...and costs
        set_optimizer_attribute(model, "max_iter", 4000) # The solver can take a LOT of iterations before it finds an optimal solution
    end
end