using JuMP
using Ipopt
using IR
using StaticArrays

include("memoization.jl")

# All I want to store is the most recent incumbent reputations and payoffs for
# Float64 and Dual types, then access them instead of calling when appropriate.

# Computation graph/list:
# - Value: Depends on [, fixed parameters]
# - Red: rem, rpm, [, red_strategy]
# - Blue: bem, bpm [, blue_strategy]
# - Players: Red, Blue
# - Judge: jem, jpm [, norm]
# - IR: Players, Judge [, prop_red]
# - IP: Players, IR [, prop_red, utilities]
# - MR: Players, Judge, IR [, prop_red]
# - MP: Players, MR [, prop_red, utilities]
# - ESS: IP, MP
# - Objective: IP


# The parameters need to be declared first because global variables 🤪
parameters = (
    norm = iNorm(191),
    red_strategy = iStrategy(12),
    blue_strategy = iStrategy(12),
    benefit_sum_max = 4, # Sum of each group's benefit
    cost_sum_min = 2, # Sum of each group's costs
    em_bound = 0.5,
    pm_bound = 0.4,
    domain_eps = 0.001,
)

all_strategies = iStrategy.(0:15)
red_mutant_strategies =
    SVector{15,eltype(all_strategies)}(setdiff(all_strategies, [parameters.red_strategy]))
blue_mutant_strategies =
    SVector{15,eltype(all_strategies)}(setdiff(all_strategies, [parameters.blue_strategy]))
# red_mutant_strategies = Vector{eltype(all_strategies)}(setdiff(all_strategies, [parameters.red_strategy]))
# blue_mutant_strategies = Vector{eltype(all_strategies)}(setdiff(all_strategies, [parameters.blue_strategy]))

function jump_stationary_incumbent_reputations(
    jem,
    jpm1::T1,
    jpm2::T1,
    jpm3::T1,
    rem,
    rpm1::T2,
    rpm2::T2,
    bem,
    bpm1::T3,
    bpm2::T3,
    prop_red;
    p,
) where {T1<:Real,T2<:Real,T3<:Real}
    jpm = SVector{3,T1}(jpm1, jpm2, jpm3)
    rpm = SVector{2,T2}(rpm1, rpm2)
    bpm = SVector{2,T3}(bpm1, bpm2)
    judge = Agent(p.norm, jem, jpm)
    red = Agent(p.red_strategy, rem, rpm)
    blue = Agent(p.blue_strategy, bem, bpm)
    stationary_incumbent_reputations(judge, red, blue, prop_red)
end

memoized_incumbent_reps =
    memoize((args...) -> jump_stationary_incumbent_reputations(args...; p = parameters), 2);

function jump_incumbent_payoffs(
    jem,
    jpm1::T1,
    jpm2::T1,
    jpm3::T1,
    rem,
    rpm1::T2,
    rpm2::T2,
    bem,
    bpm1::T3,
    bpm2::T3,
    prop_red,
    ut1::T4,
    ut2::T4,
    ut3::T4,
    ut4::T4;
    p,
) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}
    rpm = SVector{2,T2}(rpm1, rpm2)
    bpm = SVector{2,T3}(bpm1, bpm2)
    utilities = SVector{4,T4}(ut1, ut2, ut3, ut4)
    red = Agent(p.red_strategy, rem, rpm)
    blue = Agent(p.blue_strategy, bem, bpm)
    # Memoize v
    # jpm = SVector{3, T1}(jpm1, jpm2, jpm3)
    # judge = Agent(p.norm, jem, jpm)
    red_rep = memoized_incumbent_reps[1](
        jem,
        jpm1::T1,
        jpm2::T1,
        jpm3::T1,
        rem,
        rpm1::T2,
        rpm2::T2,
        bem,
        bpm1::T3,
        bpm2::T3,
        prop_red,
    )
    blue_rep = memoized_incumbent_reps[2](
        jem,
        jpm1::T1,
        jpm2::T1,
        jpm3::T1,
        rem,
        rpm1::T2,
        rpm2::T2,
        bem,
        bpm1::T3,
        bpm2::T3,
        prop_red,
    )
    # red_rep, blue_rep = stationary_incumbent_reputations(judge, red, blue, prop_red)
    # Memoize ^
    return incumbent_payoffs(red, blue, red_rep, blue_rep, prop_red, utilities)
end

jip1(xs...) = jump_incumbent_payoffs(xs...)[1]
jip2(xs...) = jump_incumbent_payoffs(xs...)[2]

function jump_ess_constraints(
    jem,
    jpm1::T1,
    jpm2::T1,
    jpm3::T1,
    rem,
    rpm1::T2,
    rpm2::T2,
    bem,
    bpm1::T3,
    bpm2::T3,
    prop_red,
    ut1::T4,
    ut2::T4,
    ut3::T4,
    ut4::T4;
    p,
) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}
    jpm = SVector{3,T1}(jpm1, jpm2, jpm3)
    rpm = SVector{2,T2}(rpm1, rpm2)
    bpm = SVector{2,T3}(bpm1, bpm2)
    utilities = SVector{4,T4}(ut1, ut2, ut3, ut4)
    judge = Agent(p.norm, jem, jpm)
    red = Agent(p.red_strategy, rem, rpm)
    blue = Agent(p.blue_strategy, bem, bpm)

    red_rep, blue_rep = stationary_incumbent_reputations(judge, red, blue, prop_red)
    incumbent_payoff_values =
        incumbent_payoffs(red, blue, red_rep, blue_rep, prop_red, utilities)
    _cons = map(all_strategies) do mutant_strategy
        red_mutant = Agent(mutant_strategy, rem, rpm)
        blue_mutant = Agent(mutant_strategy, bem, rpm)
        red_mutant_rep, blue_mutant_rep = stationary_mutant_reputations(
            judge,
            red_mutant,
            blue_mutant,
            red_rep,
            blue_rep,
            prop_red,
        )
        mutant_payoff_values = mutant_payoffs(
            red,
            blue,
            red_mutant,
            blue_mutant,
            red_rep,
            blue_rep,
            red_mutant_rep,
            blue_mutant_rep,
            prop_red,
            utilities,
        )
        incumbent_payoff_values .- mutant_payoff_values
    end
    T = eltype(eltype(_cons))
    return reinterpret(reshape, T, _cons)
end





function construct_model(p)
    model = Model(Ipopt.Optimizer)

    @variable(model, 0.5 <= prop_red <= 0.9) # Size of the majority
    @variable(model, p.domain_eps <= utilities[1:4] <= Inf) # Benefits and costs

    @variable(model, p.domain_eps <= judge_em <= p.em_bound) # Rates of execution mistakes of judge
    @variable(model, p.domain_eps <= red_em <= p.em_bound) # Rates of execution mistakes of red agents
    @variable(model, p.domain_eps <= blue_em <= p.em_bound) # Rates of execution mistakes of blue agents

    @variable(model, p.domain_eps <= judge_pm[1:3] <= p.pm_bound) # Rates of perception mistakes of judge
    @variable(model, p.domain_eps <= red_pm[1:2] <= p.pm_bound) # Rates of perception mistakes of agents
    @variable(model, p.domain_eps <= blue_pm[1:2] <= p.pm_bound) # Rates of perception mistakes of agents

    memoized_ess_constraints = memoize((args...) -> cons(args...; p), 32)
    xs = judge_em,
    judge_pm...,
    red_em,
    red_pm...,
    blue_em,
    blue_pm...,
    prop_red,
    utilities...
    for i = 0:15
        red_name = Symbol(string("red_ess_", i))
        blue_name = Symbol(string("red_ess_", i))
        @show red_name
        @eval begin
            function $(red_name)(xs...)
                return memoized_ess_constraints[$(2 * i + 1)](xs...)
            end
            register(
                $model,
                Symbol(string("red_ess_", $i)),
                15,
                $(red_name);
                autodiff = true,
            )
            @NLconstraint($model, $(blue_name)($xs...) >= $p.domain_eps)
        end
        @show blue_name
        @eval begin
            function $(blue_name)(xs...)
                return memoized_ess_constraints[$(2 * i + 2)](xs...)
            end
            register(
                $model,
                Symbol(string("blue_ess_", $i)),
                15,
                $(blue_name);
                autodiff = true,
            )
            @NLconstraint($model, $(red_name)($xs...) >= $p.domain_eps)
        end

    end
    return model
end

model = construct_model(parameters)

# memoized_incumbent_payoffs = memoize((args...) -> jump_incumbent_payoffs(               args...; p=parameters), 2);
register(model, :red_incumbent_reputation, 11, memoized_incumbent_reps[1]; autodiff = true)
register(model, :blue_incumbent_reputation, 11, memoized_incumbent_reps[2]; autodiff = true)
register(model, :red_incumbent_payoffs, 15, jip1; autodiff = true)
register(model, :blue_incumbent_payoffs, 15, jip2; autodiff = true)
register(model, :red_mutant_payoffs, 15, jmp1; autodiff = true)
register(model, :blue_mutant_payoffs, 15, jmp2; autodiff = true)


# Run:
# ?memoized_incumbent_reps    Vector{var"#259#262"{Int64, var"#foo_i#260"{var"#271#272"}}}
# ?memoized_incumbent_payoffs Vector{var"#259#262"{Int64, var"#foo_i#260"{var"#273#274"}}}

# Note how both are Vector{var"#x#x+3"{Int64, var}}

memoized_incumbent_reps[1](0.01, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.9)
memoized_incumbent_reps[2](0.01, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.9)
jmp1(0.01, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.9, 2, 2, 1, 1)
a = jump_ess_constraints(
    0.01,
    0.00,
    0.00,
    0.00,
    0.01,
    0.00,
    0.00,
    0.01,
    0.00,
    0.00,
    0.9,
    2,
    2,
    1,
    1;
    p = parameters,
)

red_ess_12(
    0.01,
    0.00,
    0.00,
    0.00,
    0.01,
    0.00,
    0.00,
    0.01,
    0.00,
    0.00,
    0.9,
    2.0,
    2.0,
    1.0,
    1.0,
)
