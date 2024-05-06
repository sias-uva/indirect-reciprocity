# Reputations
function stationary_incumbent_reputations(
    judge::Judge, majority::Player, minority::Player, prop_majority
)
    prop_maj = prop_majority # Rename for brevity
    JR(x, y) = judge(x, y, majority(x, y))
    JB(x, y) = judge(x, y, minority(x, y))
    A = SA[
        prop_maj*(JR(1, 1) - JR(1, 0)) (1 - prop_maj)*(JR(0, 1) - JR(0, 0))
        prop_maj*(JB(0, 1) - JB(0, 0)) (1 - prop_maj)*(JB(1, 1) - JB(1, 0))
    ]
    b = SA[lerp(SA[JR(0, 0), JR(1, 0)], prop_maj), lerp(SA[JB(1, 0), JB(0, 0)], prop_maj)]
    return -(A - I) \ b
end

function stationary_mutant_reputations(
    judge, majority_mutant, minority_mutant, majority_rep, minority_rep, prop_majority
)
    JR(x, y) = judge(x, y, majority_mutant(x, y))
    JB(x, y) = judge(x, y, minority_mutant(x, y))
    return SA[
        lerp(
            SA[
                lerp(SA[JR(0, 0), JR(0, 1)], minority_rep),
                lerp(SA[JR(1, 0), JR(1, 1)], majority_rep),
            ],
            prop_majority,
        ),
        lerp(
            SA[
                lerp(SA[JB(1, 0), JB(1, 1)], minority_rep),
                lerp(SA[JB(0, 0), JB(0, 1)], majority_rep),
            ],
            prop_majority,
        ),
    ]
end

function stationary_reputations(
    judge, majority, minority, majority_mutant, minority_mutant, prop_majority
)
    majority_rep, minority_rep = stationary_incumbent_reputations(
        judge, majority, minority, prop_majority
    )
    majority_mutant_rep, minority_mutant_rep = stationary_mutant_reputations(
        judge, majority_mutant, minority_mutant, majority_rep, minority_rep, prop_majority
    )
    return majority_rep, minority_rep, majority_mutant_rep, minority_mutant_rep
end
