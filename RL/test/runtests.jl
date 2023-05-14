using RL
using Test
using StaticArrays

function test_policy(policy, expected_outputs)
    for (expected_output, input) in zip(expected_outputs, Iterators.product(false:true, false:true))
        @test evaluate(policy, SA[input...]) == expected_output
    end
end

@testset "RL" begin
    @testset "Evaluating policies" begin
        @testset "AllC" begin
            all_coop = SArray{Tuple{2,2,2}}(reshape(vcat(zeros(4), ones(4)), 2,2,2))
            test_policy(all_coop, trues(4))
        end
        @testset "Disc" begin
            disc = SArray{Tuple{2,2,2}}(reshape(SA[1,1,0,0,0,0,1,1.0],2,2,2))
            test_policy(disc, [false, false, true, true])
        end
        @testset "AllD" begin
            all_defect = @SArray zeros(2,2,2)
            test_policy(all_defect, falses(4))
        end
        @testset "Inverse Disc" begin
            inv_disc = SArray{Tuple{2,2,2}}(reshape(1 .- SA[1,1,0,0,0,0,1,1.0],2,2,2))
            test_policy(inv_disc, true .- [false, false, true, true])
        end
    end

    @testset "Learning" begin
        # Learning is deterministic so is the right thing learned in each case?
        # Biggest problem is that there's a segfault/crash for some reason oops
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

