using StaticArrays # Data structure
using JuMP # Constrained programming
import Ipopt # NLP solver
using TinyIR
using Printf

include("optimised.jl")

# Globals:
## Social rules
norm::SVector{8, Bool} = SA[1, 0, 0, 0, 1, 1, 1, 1]
red_strategy::SVector{4, Bool}  = SA[0, 0, 1, 1]
blue_strategy::SVector{4, Bool} = SA[0, 0, 1, 1]
benefit_sum_max::Int = 4 # Sum of each group's benefit
cost_sum_min::Int = 2 # Sum of each group's costs

# Solver options:
domain_eps::Float64 = 0.00001

begin
    # Initialise model with Ipopt
    model = Model(Ipopt.Optimizer)

    # Register our custom objective function and auxiliary functions useful for constraints
    register(model, :avg_payoffs, 15, avg_payoffs; autodiff=true)
    register(model, :payoff_red, 15, payoff_red; autodiff=true)
    register(model, :payoff_blue, 15, payoff_blue; autodiff=true)

    # Add decision variables and their domains
    @variable(model, 0.5 <= pR <= 0.9) # Size of the majority
    @variable(model, domain_eps <= utilities[1:4] <= Inf) # Benefits and costs
    @variable(model, domain_eps <= mistakes[1:10] <= 0.05, start = 0.01) # Rates of mistakes of judge and agents
    vars = (pR, utilities..., mistakes...) # Put them all in a single variable

    # Add our non-linear objective function
    @NLobjective(model, Max, avg_payoffs(vars...)) # Maximise average payoffs...

    # ...subject to the constraint that the incumbent payoffs are larger or equal to
    # any possible mutant.
    println("Registering constraints 0 to 15")
    red_i = evalpoly(2, red_strategy)
    blue_i = evalpoly(2, blue_strategy)
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

    
    # @NLconstraint(model, payoff_red(vars...) >= 0.9 * payoff_blue(vars...))
    # @NLconstraint(model, payoff_blue(vars...) >= 0.9 * payoff_red(vars...))

    # Constrain the utilities (although some are superfluous)
    begin
        @NLconstraint(model, 0 <= utilities[1] + utilities[2] <= benefit_sum_max)
        @NLconstraint(model, utilities[3] + utilities[4] >= cost_sum_min)
        @NLconstraint(model, utilities[1] == utilities[2]) # Groups must have the same benefits...
        @NLconstraint(model, utilities[3] == utilities[4]) # ...and costs
        set_optimizer_attribute(model, "max_iter", 6000) # The solver can take a LOT of iterations before it finds an optimal solution
    end
end

optimize!(model)

summarise_solution(model, vars)
both_payoffs(value.(vars)...)

ESS_constraint(value.(vars)...)

payoff_red(value.(vars)...)
payoff_red_mutant_13(value.(vars)...)

payoff_blue(value.(vars)...)
payoff_blue_mutant_13(value.(vars)...)