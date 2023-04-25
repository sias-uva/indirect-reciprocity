using StaticArrays
using TinyIR
# using JET
# using Profile
# using PProf
# using Cthulhu

norm = SA[1, 1, 0, 0, 1, 1, 1, 1]
red_strategy = SA[0, 0, 1, 1]
blue_strategy = SA[0, 0, 1, 1]
benefit_sum_max = 4
cost_sum_min = 2

# Solver arguments:
## IR model parameters:
proportion_incumbents_red = 0.9
utils = (2, 2, 1, 1)
## Mistake probabilities
judge_mistakes = (0, (0, 0, 0))
red_mistakes = (0.01, (0, 0)) 
blue_mistakes = (0.01, (0, 0))

judge = (norm, judge_mistakes)
red = (red_strategy, red_mistakes)
blue = (blue_strategy, blue_mistakes)

@benchmark is_ESS($judge, $red, $blue, $proportion_incumbents_red, $utils)

@profview for _ in 1:1000 is_ESS(judge, red, blue, proportion_incumbents_red, utils) end 

# @report_call is_ESS(judge, red, blue, proportion_incumbents_red, utils)
# @report_opt is_ESS(judge, red, blue, proportion_incumbents_red, utils)
# @benchmark is_ESS($judge, $red, $blue, $proportion_incumbents_red, $utils)

Profile.clear()
Profile.Allocs.@profile sample_rate=1 is_ESS(judge, red, blue, proportion_incumbents_red, utils)
PProf.Allocs.pprof(from_c=false)

@report_opt(is_ESS(judge, red, blue, proportion_incumbents_red, utils))