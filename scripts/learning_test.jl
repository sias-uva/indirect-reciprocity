using IR
using RL
using StaticArrays

la1 = LearningAgent(1, iNorm(195), 0.01, zeros(SVector{2,Float64}), true, 0.05)
info = SA[1,1]
@code_warntype la1(info)