using JuMP
using Ipopt
using IR
using StaticArrays

include("memoization.jl")

# All I want to store is the most recent incumbent reputations and payoffs for
# Float64 and Dual types, then access them instead of calling when appropriate.

# Computation graph/list:
# - Value: Depends on [, fixed parameters]
# - Majority: maj_em, maj_pm, [, majority_strategy]
# - Blue: min_em, min_pm [, minority_strategy]
# - Players: Majority, Blue
# - Judge: judge_em, judge_pm [, norm]
# - IR: Players, Judge [, prop_majority]
# - IP: Players, IR [, prop_majority, utilities]
# - MR: Players, Judge, IR [, prop_majority]
# - MP: Players, MR [, prop_majority, utilities]
# - ESS: IP, MP
# - Objective: IP

# The parameters need to be declared first because global variables 🤪
parameters = (
    norm=iNorm(191),
    majority_strategy=iStrategy(12),
    minority_strategy=iStrategy(12),
    benefit_sum_max=4, # Sum of each group's benefit
    cost_sum_min=2, # Sum of each group's costs
    em_bound=0.5,
    pm_bound=0.4,
    domain_eps=0.001,
)

all_strategies = iStrategy.(0:15)
majority_mutant_strategies = SVector{15,eltype(all_strategies)}(
    setdiff(all_strategies, [parameters.majority_strategy])
)
minority_mutant_strategies = SVector{15,eltype(all_strategies)}(
    setdiff(all_strategies, [parameters.minority_strategy])
)
# majority_mutant_strategies = Vector{eltype(all_strategies)}(setdiff(all_strategies, [parameters.majority_strategy]))
# minority_mutant_strategies = Vector{eltype(all_strategies)}(setdiff(all_strategies, [parameters.minority_strategy]))

function jump_stationary_incumbent_reputations(
    judge_em,
    jpm1::T1,
    jpm2::T1,
    jpm3::T1,
    maj_em,
    rpm1::T2,
    rpm2::T2,
    min_em,
    bpm1::T3,
    bpm2::T3,
    prop_majority;
    p,
) where {T1<:Real,T2<:Real,T3<:Real}
    judge_pm = SVector{3,T1}(jpm1, jpm2, jpm3)
    maj_pm = SVector{2,T2}(rpm1, rpm2)
    min_pm = SVector{2,T3}(bpm1, bpm2)
    judge = Agent(p.norm, judge_em, judge_pm)
    majority = Agent(p.majority_strategy, maj_em, maj_pm)
    minority = Agent(p.minority_strategy, min_em, min_pm)
    return stationary_incumbent_reputations(judge, majority, minority, prop_majority)
end

memoized_incumbent_reps = memoize(
    (args...) -> jump_stationary_incumbent_reputations(args...; p=parameters), 2
);

function jump_incumbent_payoffs(
    judge_em,
    jpm1::T1,
    jpm2::T1,
    jpm3::T1,
    maj_em,
    rpm1::T2,
    rpm2::T2,
    min_em,
    bpm1::T3,
    bpm2::T3,
    prop_majority,
    ut1::T4,
    ut2::T4,
    ut3::T4,
    ut4::T4;
    p,
) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}
    maj_pm = SVector{2,T2}(rpm1, rpm2)
    min_pm = SVector{2,T3}(bpm1, bpm2)
    utilities = SVector{4,T4}(ut1, ut2, ut3, ut4)
    majority = Agent(p.majority_strategy, maj_em, maj_pm)
    minority = Agent(p.minority_strategy, min_em, min_pm)
    # Memoize v
    # judge_pm = SVector{3, T1}(jpm1, jpm2, jpm3)
    # judge = Agent(p.norm, judge_em, judge_pm)
    majority_rep = memoized_incumbent_reps[1](
        judge_em,
        jpm1::T1,
        jpm2::T1,
        jpm3::T1,
        maj_em,
        rpm1::T2,
        rpm2::T2,
        min_em,
        bpm1::T3,
        bpm2::T3,
        prop_majority,
    )
    minority_rep = memoized_incumbent_reps[2](
        judge_em,
        jpm1::T1,
        jpm2::T1,
        jpm3::T1,
        maj_em,
        rpm1::T2,
        rpm2::T2,
        min_em,
        bpm1::T3,
        bpm2::T3,
        prop_majority,
    )
    # majority_rep, minority_rep = stationary_incumbent_reputations(judge, majority, minority, prop_majority)
    # Memoize ^
    return incumbent_payoffs(
        majority, minority, majority_rep, minority_rep, prop_majority, utilities
    )
end

jip1(xs...) = jump_incumbent_payoffs(xs...)[1]
jip2(xs...) = jump_incumbent_payoffs(xs...)[2]

function jump_ess_constraints(
    judge_em,
    jpm1::T1,
    jpm2::T1,
    jpm3::T1,
    maj_em,
    rpm1::T2,
    rpm2::T2,
    min_em,
    bpm1::T3,
    bpm2::T3,
    prop_majority,
    ut1::T4,
    ut2::T4,
    ut3::T4,
    ut4::T4;
    p,
) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}
    judge_pm = SVector{3,T1}(jpm1, jpm2, jpm3)
    maj_pm = SVector{2,T2}(rpm1, rpm2)
    min_pm = SVector{2,T3}(bpm1, bpm2)
    utilities = SVector{4,T4}(ut1, ut2, ut3, ut4)
    judge = Agent(p.norm, judge_em, judge_pm)
    majority = Agent(p.majority_strategy, maj_em, maj_pm)
    minority = Agent(p.minority_strategy, min_em, min_pm)

    majority_rep, minority_rep = stationary_incumbent_reputations(
        judge, majority, minority, prop_majority
    )
    incumbent_payoff_values = incumbent_payoffs(
        majority, minority, majority_rep, minority_rep, prop_majority, utilities
    )
    _cons = map(all_strategies) do mutant_strategy
        majority_mutant = Agent(mutant_strategy, maj_em, maj_pm)
        minority_mutant = Agent(mutant_strategy, min_em, maj_pm)
        majority_mutant_rep, minority_mutant_rep = stationary_mutant_reputations(
            judge,
            majority_mutant,
            minority_mutant,
            majority_rep,
            minority_rep,
            prop_majority,
        )
        mutant_payoff_values = mutant_payoffs(
            majority,
            minority,
            majority_mutant,
            minority_mutant,
            majority_rep,
            minority_rep,
            majority_mutant_rep,
            minority_mutant_rep,
            prop_majority,
            utilities,
        )
        incumbent_payoff_values .- mutant_payoff_values
    end
    T = eltype(eltype(_cons))
    return reinterpret(reshape, T, _cons)
end

function construct_model(p)
    model = Model(Ipopt.Optimizer)

    @variable(model, 0.5 <= prop_majority <= 0.9) # Size of the majority
    @variable(model, p.domain_eps <= utilities[1:4] <= Inf) # Benefits and costs

    @variable(model, p.domain_eps <= judge_em <= p.em_bound) # Rates of execution mistakes of judge
    @variable(model, p.domain_eps <= majority_em <= p.em_bound) # Rates of execution mistakes of majority agents
    @variable(model, p.domain_eps <= minority_em <= p.em_bound) # Rates of execution mistakes of minority agents

    @variable(model, p.domain_eps <= judge_pm[1:3] <= p.pm_bound) # Rates of perception mistakes of judge
    @variable(model, p.domain_eps <= majority_pm[1:2] <= p.pm_bound) # Rates of perception mistakes of agents
    @variable(model, p.domain_eps <= minority_pm[1:2] <= p.pm_bound) # Rates of perception mistakes of agents

    memoized_ess_constraints = memoize((args...) -> cons(args...; p), 32)
    xs = judge_em,
    judge_pm...,
    majority_em,
    majority_pm...,
    minority_em,
    minority_pm...,
    prop_majority,
    utilities...
    for i in 0:15
        majority_name = Symbol(string("majority_ess_", i))
        minority_name = Symbol(string("majority_ess_", i))
        @show majority_name
        @eval begin
            function $(majority_name)(xs...)
                return memoized_ess_constraints[$(2 * i + 1)](xs...)
            end
            register(
                $model,
                Symbol(string("majority_ess_", $i)),
                15,
                $(majority_name);
                autodiff=true,
            )
            @NLconstraint($model, $(minority_name)($xs...) >= $p.domain_eps)
        end
        @show minority_name
        @eval begin
            function $(minority_name)(xs...)
                return memoized_ess_constraints[$(2 * i + 2)](xs...)
            end
            register(
                $model,
                Symbol(string("minority_ess_", $i)),
                15,
                $(minority_name);
                autodiff=true,
            )
            @NLconstraint($model, $(majority_name)($xs...) >= $p.domain_eps)
        end
    end
    return model
end

model = construct_model(parameters)

# memoized_incumbent_payoffs = memoize((args...) -> jump_incumbent_payoffs(               args...; p=parameters), 2);
register(
    model, :majority_incumbent_reputation, 11, memoized_incumbent_reps[1]; autodiff=true
)
register(
    model, :minority_incumbent_reputation, 11, memoized_incumbent_reps[2]; autodiff=true
)
register(model, :majority_incumbent_payoffs, 15, jip1; autodiff=true)
register(model, :minority_incumbent_payoffs, 15, jip2; autodiff=true)
register(model, :majority_mutant_payoffs, 15, jmp1; autodiff=true)
register(model, :minority_mutant_payoffs, 15, jmp2; autodiff=true)

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
    p=parameters,
)

majority_ess_12(
    0.01, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.9, 2.0, 2.0, 1.0, 1.0
)
