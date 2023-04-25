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

function jump_stationary_incumbent_reputations(jem, jpm1::T1, jpm2::T1, jpm3::T1, rem, rpm1::T2, rpm2::T2, bem, bpm1::T3, bpm2::T3, prop_red; p) where {T1<:Real, T2<:Real, T3<:Real}
    jpm = SVector{3, T1}(jpm1, jpm2, jpm3)
    rpm = SVector{2, T2}(rpm1, rpm2)
    bpm = SVector{2, T3}(bpm1, bpm2)
    judge = Agent(p.norm, jem, jpm)
    red = Agent(p.red_strategy, rem, rpm)
    blue = Agent(p.blue_strategy, bem, bpm)
    stationary_incumbent_reputations(judge, red, blue, prop_red)
end

function jump_incumbent_payoffs(jem, jpm1::T1, jpm2::T1, jpm3::T1, rem, rpm1::T2, rpm2::T2, bem, bpm1::T3, bpm2::T3, prop_red, ut1::T4, ut2::T4, ut3::T4, ut4::T4; p) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real}
    rpm = SVector{2, T2}(rpm1, rpm2)
    bpm = SVector{2, T3}(bpm1, bpm2)
    utilities = SVector{4, T4}(ut1, ut2, ut3, ut4)
    red = Agent(p.red_strategy, rem, rpm)
    blue = Agent(p.blue_strategy, bem, bpm)
    # Memoize v
    jpm = SVector{3, T1}(jpm1, jpm2, jpm3)
    judge = Agent(p.norm, jem, jpm)
    println("")
    println("CALLING IP")
    # @show judge red blue prop_red
    @show stationary_incumbent_reputation(judge, red, blue, prop_red)
    @show stationary_incumbent_reputation
    red_rep, blue_rep = stationary_incumbent_reputation(judge, red, blue, prop_red)
    @show red_rep blue_rep
    # Memoize ^
    ip = incumbent_payoffs(red, blue, red_rep, blue_rep, prop_red, utilities)
    if isnothing(ip)
        @show red blue red_rep blue_rep
    end
    return ip
end

parameters = (
    norm = iNorm(191),
    red_strategy = iStrategy(12),
    blue_strategy = iStrategy(12),
    benefit_sum_max = 4, # Sum of each group's benefit
    cost_sum_min = 2, # Sum of each group's costs
    em_bound = 0.5,
    pm_bound = 0.4,
)

model = Model(Ipopt.Optimizer);
memoized_incumbent_reps =    memoize((args...) -> jump_stationary_incumbent_reputations(args...; p=parameters), 2);
memoized_incumbent_payoffs = memoize((args...) -> jump_incumbent_payoffs(               args...; p=parameters), 2);
register(model, :red_incumbent_reputation,  11, memoized_incumbent_reps[1]; autodiff = true)
register(model, :blue_incumbent_reputation, 11, memoized_incumbent_reps[2]; autodiff = true)
register(model, :red_incumbent_payoffs,  15, memoized_incumbent_payoffs[1]; autodiff = true) # Doesn't recalculate primal/dual cache??
register(model, :blue_incumbent_payoffs, 15, memoized_incumbent_payoffs[2]; autodiff = true)

# Run:
# ?memoized_incumbent_reps    Vector{var"#259#262"{Int64, var"#foo_i#260"{var"#271#272"}}}
# ?memoized_incumbent_payoffs Vector{var"#259#262"{Int64, var"#foo_i#260"{var"#273#274"}}}

# Note how both are Vector{var"#x#x+3"{Int64, var}}

memoized_incumbent_reps[1](0.01, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.9)
memoized_incumbent_reps[2](0.01, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.9)
memoized_incumbent_payoffs[1](0.01, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.9, 2.0, 2.0, 1.0, 1.0)
memoized_incumbent_payoffs[2](0.01, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.9, 2.0, 2.0, 1.0, 1.0)

dx0 = Dual.(0.01, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.9)

x0 = 0.00, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.9, 2, 2, 1, 1
jump_incumbent_payoffs(0.00, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.9, 2, 2, 1, 1; p=parameters)
jump_incumbent_payoffs(x0...; p=parameters)
jump_incumbent_payoffs(dx0...; p=parameters)
ForwardDiff.jacobian(
    x -> memoized_incumbent_reps[1](x...),
    [x0[1:11]...],
    ForwardDiff.JacobianConfig(x -> memoized_incumbent_reps[1](x...), [x0[1:11]...]),
    Val{false}()
)
memoized_incumbent_payoffs[1](x0...)
