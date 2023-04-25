using IR
using StaticArrays

jem = 0.01
jpm = SA[0.0, 0.0, 0.0]
rem = 0.01
rpm = SA[0.0, 0.0]
bem = 0.01
bpm = SA[0.0, 0.0]

pR = 0.9

utilities = SA[2,2,1,1]


function find_ESS(jem, jpm, rem, rpm, bem, bpm, pR, utilities)
    which = 0
    res = falses(65536)
    for norm_i in 0:255
        norm = iNorm(norm_i)
        judge = Agent(norm, jem, jpm)
        for rs_i in 0:15
            red_strat = iStrategy(rs_i)
            red = Agent(red_strat, rem, rpm)
            for bs_i in 0:15
                which += 1
                blue_strat = iStrategy(bs_i)
                blue = Agent(blue_strat, bem, bpm)
                res[which] = is_ESS(judge, red, blue, pR, utilities)
            end
        end
    end
    return res
end

ess = find_ESS(jem, jpm, rem, rpm, bem, bpm, pR, utilities)
count(ess)