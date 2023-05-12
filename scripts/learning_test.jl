using IR
using RL
using StaticArrays

la1 = LearningAgent(1, iNorm(195), 0.01, zeros(SVector{2,Float64}), true, 0.05)
info = SA[1,1]

reduce(argmax, iNorm(195); dims=3)
argmax(iNorm(195); dims=3)
maximum(iNorm(192); dims=3)
reduce(hcat, iNorm(192); dims=3)


la1(info)

@code_warntype la1(info)

pol1 = rand(SArray{Tuple{2,2,2},Float64})

reduce(whichmax, pol1; dims=3)