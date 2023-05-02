using IR
using StaticArrays
using Printf
using JuMP
using Ipopt

# Globals:
## Social rules
norm::Norm = iNorm(195)
red_strategy::Strategy = iStrategy(12)
blue_strategy::Strategy = iStrategy(12)
benefit_sum_max::Int = 4 # Sum of each group's benefit
cost_sum_min::Int = 2 # Sum of each group's costs
em_bound::Float64 = 0.5
pm_bound::Float64 = 0.5

# Solver options:
domain_eps::Float64 = 0.0001



begin
    include("jump_integration.jl")
    # Initialise model with Ipopt
    model = Model(Ipopt.Optimizer)

    # Register our custom objective function and auxiliary functions useful for constraints
    register(model, :avg_payoffs, 17, avg_payoffs; autodiff = true)
    register(model, :payoff_red, 17, payoff_red; autodiff = true)
    register(model, :payoff_blue, 17, payoff_blue; autodiff = true)
    register(model, :stationary_reps, 17, stationary_reps; autodiff = true)
    register(model, :stationary_reps2, 17, stationary_reps2; autodiff = true)

    # Add decision variables and their domains
    @variable(model, 0.5 <= pR <= 0.9) # Size of the majority

    @variable(model, domain_eps <= utilities[1:4] <= Inf) # Benefits and costs

    @variable(model, domain_eps <= judge_em <= em_bound) # Rates of execution mistakes of judge
    @variable(model, domain_eps <= red_em <= em_bound) # Rates of execution mistakes of red agents
    @variable(model, domain_eps <= blue_em <= em_bound) # Rates of execution mistakes of blue agents

    @variable(model, domain_eps <= judge_pm[1:3] <= pm_bound) # Rates of perception mistakes of judge
    @variable(model, domain_eps <= red_pm[1:2] <= pm_bound) # Rates of perception mistakes of agents
    @variable(model, domain_eps <= blue_pm[1:2] <= pm_bound) # Rates of perception mistakes of agents

    @variable(model, 0 <= reputations[1:2] <= 1)

    # @variable(model, domain_eps <= mistakes[1:10] <= 0.005, start = 0.01) # Rates of mistakes of judge and agents
    vars = (
        pR,
        utilities...,
        judge_em,
        judge_pm...,
        red_em,
        red_pm...,
        blue_em,
        blue_pm...,
        reputations...,
    ) # Put them all in a single variable

    # Add our non-linear objective function
    @NLobjective(model, Max, avg_payoffs(vars...)) # Maximise average payoffs...

    # ...subject to the constraint that the incumbent payoffs are larger or equal to
    # any possible mutant.
    println("Registering constraints 0 to 15")
    red_i = @evalpoly(2, red_strategy...)
    blue_i = @evalpoly(2, blue_strategy...)
    @show red_i blue_i
    for i = 0:15
        print(i)
        # We do a really nasty metaprogramming trick to make this work. Please don't do this ever.
        red_name = Symbol(string("payoff_red_mutant_", i))
        blue_name = Symbol(string("payoff_blue_mutant_", i))
        @eval begin
            if $red_i != $i
                register(
                    model,
                    Symbol(string("payoff_red_mutant_", $i)),
                    17,
                    $(red_name);
                    autodiff = true,
                )
                @NLconstraint(
                    model,
                    payoff_red(vars...) >= domain_eps + $(red_name)(vars...)
                )
                print(".")
            else
                print("r")
            end
            if $blue_i != $i
                register(
                    model,
                    Symbol(string("payoff_blue_mutant_", $i)),
                    17,
                    $(blue_name);
                    autodiff = true,
                )
                @NLconstraint(
                    model,
                    payoff_blue(vars...) >= domain_eps + $(blue_name)(vars...)
                )
                print(".")
            else
                print("b")
            end
        end
        mod(i, 4) == 3 && println()
    end


    # @NLconstraint(model, payoff_red(vars...) >= 0.6 * payoff_blue(vars...))
    # @NLconstraint(model, payoff_blue(vars...) >= 0.6 * payoff_red(vars...))

    # Constrain reputations to conform
    @NLconstraint(model, reputations[1] >= stationary_reps(vars...))
    @NLconstraint(model, reputations[2] >= stationary_reps2(vars...))

    # Constrain the utilities (although some are superfluous)
    begin
        @NLconstraints(model, begin
            0 <= utilities[1] + utilities[2] <= benefit_sum_max
            utilities[3] + utilities[4] >= cost_sum_min
            utilities[1] == utilities[2]# Groups must have the same benefits...
            utilities[3] == utilities[4] # ...and costs
        end)
    end
    set_optimizer_attribute(model, "max_iter", 4000) # The solver can take a LOT of iterations before it finds an optimal solution
end

unset_silent(model)
optimize!(model)

summarise_solution(model, vars)
both_payoffs(value.(vars)...)

ESS_constraint(value.(vars)...)
# ESS_constraint(value(pR), (2,2,1,1)..., value.(mistakes)...)

payoff_red(value.(vars)...)
payoff_red_mutant_13(value.(vars)...)

payoff_blue(value.(vars)...)
payoff_blue_mutant_13(value.(vars)...)

(48.616 * (2^2^2) * (2^2^3)) / 1000 # ms
199 / 60
