import IR
import TinyIR
using StaticArrays


em = 0.01
pm = SA[zeros(2)...]
jpm = SA[zeros(3)...]
pR = 0.9

for i in 0:255
    ir_reps = let
        s = IR.iStrategy(12)
        red = IR.Agent(s, em, pm)
        blue = IR.Agent(s, em, pm)
        judge = IR.Agent(IR.iNorm(i), em, jpm)
        IR.stationary_incumbent_reputations(judge, red, blue, pR)
    end
    tiny_reps = let
        s = SA[0, 0, 1, 1]
        red = (s, (em, pm))
        blue = (s, (em, pm))
        judge = (SVector(IR.iNorm(i)), (em, jpm))
        TinyIR.stationary_incumbent_reputations(judge, red, blue, pR)
    end
end



norm = SA[1, 1, 0, 0, 1, 1, 1, 1]
red_strategy = 
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

@test is_ESS(judge, red, blue, proportion_incumbents_red, utils)




IR.iNorm(225)
IR.iStrategy(12)
norm::IR.Norm = SA[0, 0,0,0,0,0,0,0]
red_strategy::IR.Strategy  = SA[0, 0, 0, 0]


info = SA[1,1]
a = Agent(red_strategy, em, pm)
j = Agent(norm, em, jpm)

stationary_incumbent_reputations(j, a, a, 0.9)

# 1. Figure out new encoding
# 2. Apply new encoding
# 3. Make AUC graph