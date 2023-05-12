using Base:Fix1
using RL
using IR
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

info1 = SA[true, true]
info2 = SA[0.90, 0.90]
alpha1 = SA[0.1, 0.2]
eps1 = 0.2



i=0
j=0
for _ in 1:1000
    perceived_info = mistake(alpha1, info1) .> rand(SVector{2,Float64}) 
    intention = evaluate(pol1, perceived_info)
    execution = mistake(eps1, intention) > rand()

    prob_execution = (
        Fix1(mistake, eps1) ∘ Fix1(evaluate, pol1) ∘ Fix1(mistake, alpha1)
    )(info1)

    i += execution
    j += prob_execution > rand()
end
i
j
@btime evaluate($pol1, $info1)
@btime evaluate($pol1, $info2)