using TinyIR

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

judge1 = (norm, judge_mistakes)
red = (red_strategy, red_mistakes)
blue = (blue_strategy, blue_mistakes)

@btime TinyIR.stationary_incumbent_reputations(judge1, red, blue, proportion_incumbents_red)
@btime TinyIR.stationary_incumbent_reputations2(judge1, red, blue, proportion_incumbents_red)