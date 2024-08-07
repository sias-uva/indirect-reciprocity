using Distributions
using CairoMakie
using IR
using OrdinaryDiffEq
using StaticArrays

const Norm = SArray{NTuple{2,2},Bool}
const Strategy = SMatrix{2,2,Int}
const ReportingRule = SArray{NTuple{2,2},Bool}

# Functions for getting social rules from number encodings
IR.iNorm(i::Integer) = Norm(i >> shift & 1 != 0 for shift in 0:3)

# A report doesn't happen for the action "2", this just bypasses the entire interaction.
# This needs to be be incorporated into the maths
iReport(i::Integer) = ReportingRule(i >> shift & 1 != 0 for shift in 0:3)
function IR.iStrategy(i::Integer)
    m_out = zero(MVector{4,Int})
    for idx in 1:4
        i, r = divrem(i, 3)
        m_out[idx] = r
        i == 0 && break
    end
    return Strategy(m_out)
end

#TODO: The other way around (number from rule) and explainers

# We need a separate measure of the 2-ness of the action taken, as this just
# results in continuation i.e. du += u × ℙ(action=2).

"""
    act(s::Strategy, goodness, group; error=false)

First output represents action taken if interaction takes place, second output
is chance that agent decides to not interact.
"""
function act(s::Strategy, goodness, group; error=false)
    out = zero(SVector{2,Float64})
    inputs_product = Iterators.product((1 - goodness, goodness), (1 - group, group))
    for (Is, (go, gr)) in zip(CartesianIndices(s), inputs_product)
        is_skip = s[Is] == 2
        out =
            out +
            go * gr * (is_skip * SA[false, true] + (1 - error) * !is_skip * SA[true, false])
    end
    return out
end

function report(r::ReportingRule, action, group)
    inputs_product = Iterators.product((1 - action, action), (1 - group, group))
    return sum(zip(r, inputs_product)) do (rating, (action, gr))
        action * gr * rating
    end
end

function judge(n::Norm, goodness, rating)
    inputs_product = Iterators.product((1 - goodness, goodness), (1 - rating, rating))
    return sum(zip(n, inputs_product)) do (rating, (go, rat))
        go * rat * rating
    end
end

act(Strategy((1, 1, 2, 1)), 0.5, 0.1; error=true)
report(ReportingRule((1, 1, 0, 1)), 0.5, 0.1)

# Hence act should return 2-ness and 01-ness separately
function step!(du, u, p, t)
    @. du = -u
    for i in false:true
        p_i = i * p.prop_maj + !i * (1 - p.prop_maj) # prob of interacting as group i
        for j in false:true
            p_j = j * p.prop_maj + !j * (1 - p.prop_maj) # prob of interacting with group j
            println("")
            for i_is_good in false:true
                gi = i_is_good * u[i+1] + !i_is_good * (1 - u[i+1]) # goodness of i
                for j_is_good in false:true
                    gj = j_is_good * u[j+1] + !j_is_good * (1 - u[j+1]) # goodness of j
                    ai, si = act(p.action_rules[i+1], gj, j; error=p.pem) # i acts
                    aj, sj = act(p.action_rules[j+1], gi, i; error=p.pem) # j acts
                    rj = report(p.reporting_rules[j+1], ai, i) # j reports
                    ri = report(p.reporting_rules[i+1], aj, j) # i reports
                    du[i+1] += (p_i * p_j) * (si * u[i+1] + (1 - si) * (1 - sj) * judge(p.norm, gj, rj))
                    du[j+1] += (p_i * p_j) * (sj * u[j+1] + (1 - si) * (1 - sj) * judge(p.norm, gi, ri))
                    println("groups($(Int(i)), $(Int(j))), goodness($gi, $gj), skips($si, $sj), actions($ai, $aj)")
                end
            end
        end
    end
end

function main()
    player_execution_mistake_rate = 0.01
    player_reporting_mistake_rate = 0.01
    judge_execution_mistake_rate = 0.01
    proportion_incumbents_majority = 0.8

    majority_reporting_rule = iReport(15)
    minority_reporting_rule = iReport(15)
    majority_strat = iStrategy(15)
    minority_strat = iStrategy(15)
    norm = iNorm(9)

    p = (;
        pem=player_execution_mistake_rate,
        prm=player_reporting_mistake_rate,
        jem=judge_execution_mistake_rate,
        prop_maj=proportion_incumbents_majority,
        action_rules=[majority_strat, minority_strat],
        reporting_rules=[majority_reporting_rule, minority_reporting_rule],
        norm,
    )

    u0 = [1/4, 1/3]
    du = similar(u0)
    tspan = (0.0, 10.0)
    prob = ODEProblem(step!, u0, tspan, p)
    # step!(du, u0, p, tspan)
    ode_sol = solve(prob, Tsit5()) #; reltol=1e-8, abstol=1e-8)
    # true_solution = stationary_incumbent_reputations(judge, majority, minority, p.prop_maj)
    # ss_sol = solve(NonlinearProblem(prob), NewtonRaphson())
    # approx_eq_or_closer(ode_sol, true_solution)
    # display(true_solution)
    # display(ss_sol)
    # b_analytical = @benchmark stationary_incumbent_reputations($judge, $majority, $minority, $p.prop_maj)
    # display(b_analytical)
    # b_sciml = @benchmark solve(NonlinearProblem($prob), NewtonRaphson())
    # display(b_sciml)
    # begin
    #     resolution = (550, 500)
    #     fig = Figure(; resolution)
    #     ax = Axis(fig[1, 1];)
    #     lines!(ode_sol.t, first.(ode_sol.u))
    #     lines!(ode_sol.t, last.(ode_sol.u))
    #     hlines!.(true_solution, linestyle=:dash)
    #     fig
    # end
end

main()
