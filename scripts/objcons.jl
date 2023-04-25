using IR
using StaticArrays
using ObjConsNLPModels, Percival, NLPModels

# struct ArbitraryCallable{T} <: Function
#     iv::T
# end

# function (ac::ArbitraryCallable{T})(x) where {T}
#     obj = sum(abs2, x)
#     cons = sum(x) - ac.iv
#     return [obj, cons]
# end
# ac = ArbitraryCallable{Float64}(1)


# model = objcons_nlpmodel(ac; x0=[2.0, 2.0])

tmodel = objcons_nlpmodel(x -> [sum(abs2, x), sum(x) - 1]; x0 = [2.0, 2.0])
output = percival(tmodel)
output.solution
output.objective


struct IRSystem{T,N<:SArray{NTuple{3,2},T},S<:SArray{NTuple{2,2},T}}
    norm::N
    red_strategy::S
    blue_strategy::S
    red_mutant_strategies::Vector{S}
    blue_mutant_strategies::Vector{S}
end

function IRSystem(n, r, b)
    all_strategies = iStrategy.(0:15)
    S = typeof(r)
    mrs = Vector{S}(setdiff(all_strategies, [r]))
    mbs = Vector{S}(setdiff(all_strategies, [b]))
    IRSystem(n, r, b, mrs, mbs)
end

norm::Norm = iNorm(195)
red_strategy::Strategy = iStrategy(12)
blue_strategy::Strategy = iStrategy(12)

irs = IRSystem(norm, red_strategy, blue_strategy)

function (irs::IRSystem)(jem, jpm, rem, rpm, bem, bpm, prop_red, utilities)
    judge = Agent(irs.norm, jem, jpm)
    red = Agent(irs.red_strategy, rem, rpm)
    blue = Agent(irs.blue_strategy, bem, bpm)
    red_rep, blue_rep = stationary_incumbent_reputations(judge, red, blue, prop_red)
    incumbent_payoff_values = incumbent_payoffs(red, blue, red_rep, blue_rep, prop_red, utilities)
    _cons = map(zip(irs.red_mutant_strategies, irs.blue_mutant_strategies)) do (rms, bms)
        red_mutant = Agent(rms, rem, rpm)
        blue_mutant = Agent(bms, bem, rpm)
        red_mutant_rep, blue_mutant_rep = stationary_mutant_reputations(judge, red_mutant, blue_mutant, red_rep, blue_rep, prop_red)
        mutant_payoff_values = mutant_payoffs(red, blue, red_mutant, blue_mutant, red_rep, blue_rep, red_mutant_rep, blue_mutant_rep, prop_red, utilities)
        incumbent_payoff_values .- mutant_payoff_values
    end
    T = eltype(eltype(_cons))
    cons = reinterpret(T, _cons)
    obj = mean(incumbent_payoff_values)
    return [obj, cons...]
end

function (irs::IRSystem)(x)
    T = typeof(x[1])
    jem = x[1]
    jpm = SVector{3, T}(x[2:4])
    rem = x[5]
    rpm = SVector{2, T}(x[6:7])
    bem = x[8]
    bpm = SVector{2, T}(x[9:10])
    prop_red = x[11]
    utilities = SVector{4, T}(x[12:15])
    return irs(jem, jpm, rem, rpm, bem, bpm, prop_red, utilities)
end

x0 = [0., SA[0,0,0.]..., 0.01, SA[0,0.]..., 0.01, SA[0,0.]..., 0.8, SA[2,2,1,1]...]

constraint_parameters = (
    benefit_sum_max = 4, # Sum of each group's benefit
    cost_sum_min = 2, # Sum of each group's costs
    em_bound = 0.5,
    pm_bound = 0.4,
)

lvar = zeros(15)
uvar = let
    bounds = ones(15)
    ems = [1, 5, 8]
    pms = [2:4..., 6:7..., 9:10...]
    bounds[ems] .= constraint_parameters[:em_bound]
    bounds[pms] .= constraint_parameters[:pm_bound]
    bounds[12:15] .= Inf
    bounds
end

irs(x0)[1]
all(irs(x0)[2:end] .> 0)

model = objcons_nlpmodel(irs; x0 = x0, lvar, uvar)