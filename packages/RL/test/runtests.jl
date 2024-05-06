using IR
using RL
using Test
using StaticArrays

function test_policy(policy, expected_outputs)
    for (expected_output, input) in
        zip(expected_outputs, Iterators.product(false:true, false:true))
        @test evaluate(policy, SA[input...]) == expected_output
    end
end

@testset "RL" begin
    all_coop = SArray{Tuple{2,2,2}}(reshape(vcat(zeros(4), ones(4)), 2, 2, 2))
    disc = SArray{Tuple{2,2,2}}(reshape(SA[1, 1, 0, 0, 0, 0, 1, 1.0], 2, 2, 2))
    all_defect = @SArray zeros(2, 2, 2)
    inv_disc = SArray{Tuple{2,2,2}}(reshape(1 .- SA[1, 1, 0, 0, 0, 0, 1, 1.0], 2, 2, 2))
    @testset "Evaluating policies" begin
        @testset "AllC" test_policy(all_coop, trues(4))
        @testset "Disc" test_policy(disc, [false, false, true, true])
        @testset "AllD" test_policy(all_defect, falses(4))
        @testset "Inverse Disc" test_policy(inv_disc, true .- [false, false, true, true])
    end

    @testset "Learning" begin
        model = initialise_abm(; n_agents=2, norm=iNorm(195))
        la_maj = model.agents[1] = LearningAgent(1, disc, 0.01, SA[0.0, 0.0], true, true)
        la_min =
            model.agents[2] = LearningAgent(2, all_defect, 0.01, SA[0.0, 0.0], false, true)

        # Set la_maj's in-good-coop value to infinity, then reset it to 1
        old_policy = la_maj.policy
        learn!(la_maj, Inf, model)
        la_maj.memory = SA[1, 1, 1]
        @test old_policy == la_maj.policy # Not yet interacted, shouldn't learn

        la_maj.memory = SA[1, 1, 1]
        la_maj.has_interacted_as_donor = true
        learn!(la_maj, Inf, model)
        policy_diff = (old_policy .== la_maj.policy)
        changed_index = CartesianIndex(2, 2, 2)
        for I in CartesianIndices(old_policy)
            if I != changed_index
                @test policy_diff[I]
            else
                @test !policy_diff[I]
                @test isinf(la_maj.policy[I])
            end
        end
        la_maj.policy = setindex(la_maj.policy, 1, 8)
        # la_min learns to cooperate
        old_min_policy = la_min.policy
        la_min.has_interacted_as_donor = true
        la_min.memory = SA[true, true, false] # la_min remembers not cooperating with in-good
        # learn!(la_min, 0, ) # assume, then, that 
    end

    @testset "Stepping agents" begin
        # Initialise agents with some known strategies, step them in a specific
        # order, test whether the outcomes are what is expected each time
        # (possibly no exploration), and what is learned makes sense.
    end

    @testset "Integration test" begin
        # Test whether a typical run goes as expected by getting the average
        # policy before and after 100 steps in a system of 10 agents with 2:3
        # ratio of minority:majority.
    end
end
