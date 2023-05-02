# Reputations
function stationary_incumbent_reputations(judge::Judge, red::Player, blue::Player, prop_red)
    pR = prop_red # Rename for brevity
    JR(x, y) = judge(x, y, red(x, y))
    JB(x, y) = judge(x, y, blue(x, y))
    A = SA[
        pR*(JR(1, 1) - JR(1, 0)) (1 - pR)*(JR(0, 1) - JR(0, 0))
        pR*(JB(0, 1) - JB(0, 0)) (1 - pR)*(JB(1, 1) - JB(1, 0))
    ]
    b = SA[lerp(SA[JR(0, 0), JR(1, 0)], pR), lerp(SA[JB(1, 0), JB(0, 0)], pR)]
    return -(A - I) \ b
end

function stationary_mutant_reputations(
    judge, red_mutant, blue_mutant, red_rep, blue_rep, prop_red
)
    JR(x, y) = judge(x, y, red_mutant(x, y))
    JB(x, y) = judge(x, y, blue_mutant(x, y))
    return SA[
        lerp(
            SA[
                lerp(SA[JR(0, 0), JR(0, 1)], blue_rep),
                lerp(SA[JR(1, 0), JR(1, 1)], red_rep),
            ],
            prop_red,
        ),
        lerp(
            SA[
                lerp(SA[JB(1, 0), JB(1, 1)], blue_rep),
                lerp(SA[JB(0, 0), JB(0, 1)], red_rep),
            ],
            prop_red,
        ),
    ]
end

function stationary_reputations(judge, red, blue, red_mutant, blue_mutant, prop_red)
    red_rep, blue_rep = stationary_incumbent_reputations(judge, red, blue, prop_red)
    red_mutant_rep, blue_mutant_rep = stationary_mutant_reputations(
        judge, red_mutant, blue_mutant, red_rep, blue_rep, prop_red
    )
    return red_rep, blue_rep, red_mutant_rep, blue_mutant_rep
end
