module RL
using Reexport

export LearningAgent, initialise_abm, agent_step!, play_and_learn!, learn! # core
export whichmax, evaluate # misc

@reexport import IR: iNorm
using IR: lerp, mistake, execution_oopsie, Norm, Agent
using Agents
using StaticArrays
using Base: Fix1

whichmax(x, y) = argmax((x, y)) - 1

mutable struct LearningAgent{P,E,A,G} <: AbstractAgent
    id::Int
    policy::P
    ε::E
    α::A
    group::G
    reputation::Bool
    memory::SVector{3,Bool}
    has_interacted_as_donor::Bool
    function LearningAgent(
        id,
        policy::P,
        ε::E,
        α::A,
        group::G,
        reputation,
        memory=SVector{3,Bool}(true, true, true),
        has_interacted_as_donor=false,
    ) where {P,E,A,G}
        return new{P,E,A,G}(
            id, policy, ε, α, group, reputation, memory, has_interacted_as_donor
        )
    end
end

function evaluate(policy::P, info::SVector{2}) where {T,P<:SArray{Tuple{2,2,2},T}}
    strategy = SArray{NTuple{2,2},T,2,4}(reduce(whichmax, policy; dims=3))
    return lerp(strategy, info)
end

# agent_step! can also trigger a segfault! what joy!
function agent_step!(donor::LearningAgent, model)
    recipient_id = let
        i = rand(1:(nagents(model) - 1))
        i < donor.id ? i : i + 1
    end
    recipient = model[recipient_id]
    return play_and_learn!((donor, recipient), model)
end

function play_and_learn!((donor, recipient)::NTuple{2,LearningAgent}, model)
    # println("Interaction: $(donor.id), $(recipient.id)")
    donor.has_interacted_as_donor = true
    is_same_group = donor.group == recipient.group
    is_good = recipient.reputation
    info = SA[is_same_group, is_good]
    perceived_info = mistake(donor.α, info) .> rand(SVector{2,Float64})
    # println("Perceived: $perceived_info, index: $(perceived_info .+ 1)")
    if model.exploration_rate > rand()
        outcome = rand(Bool) # Explore
    else
        prob_coop = (Fix1(execution_oopsie, donor.ε) ∘ Fix1(evaluate, donor.policy))(
            perceived_info
        )
        outcome = prob_coop > rand() # Q-learning
    end
    # outcome = softmax(donor.policy[perceived_info .+ 1..., :])[1] > rand() # Roth-Erev learning
    judgement = model.judge(SA[is_same_group, is_good, outcome]) > rand()
    donor.reputation = judgement
    donor.memory = SA[perceived_info..., outcome]
    # Agents learn whether or not there was a donation, utilities adjusted
    # accordingly. No donation => no cost, no benefit but decay of corresponding
    # Q-value still takes place.
    # println("Learning")
    # println("Donor $(donor.id) donated to $(recipient.id)? ", outcome)
    cost = donor.group ? model.utilities[4] : model.utilities[3]
    learn!(donor, outcome * -cost, model)
    benefit = donor.group ? model.utilities[2] : model.utilities[1]
    learn!(recipient, outcome * benefit, model)
    return nothing
end

softmax(xs) = exp.(xs) / sum(exp.(xs))

function learn!(la::LearningAgent, utility, model)
    !la.has_interacted_as_donor && return nothing
    interaction = la.memory
    # println("Agent $(la.id) learned $utility from $(la.memory)")
    idx = interaction .+ 1
    ϕ = model.learning_rate
    old_policy = la.policy
    new_q = (1 - ϕ) * old_policy[idx...] + ϕ * utility # Q-learning
    linear_idx = interaction[1] + 2 * interaction[2] + 4 * interaction[3] + 1
    la.policy = setindex(old_policy, new_q, linear_idx)
    return nothing
end

# You can trigger a segfault with this function sometimes lol, the reason is
# contained within!
function initialise_abm(;
    n_agents=50,
    norm,
    majority_pm=SA[0.0, 0.0],
    majority_em=0.01,
    minority_pm=SA[0.0, 0.0],
    minority_em=0.01,
    judge_pm=SA[0.0, 0.0, 0.0],
    judge_em=0.01,
    utilities=SA[2.0, 2.0, 1.0, 1.0],
    learning_rate=0.01,
    exploration_rate=0.01,
    proportion_incumbents_majority=0.9,
)
    judge = Agent(norm, judge_em, judge_pm)
    properties = (; judge, utilities, learning_rate, exploration_rate)
    example_strategy = rand(SArray{Tuple{2,2,2},Float64,3,8})
    example_agent = LearningAgent(
        -1,
        example_strategy,
        majority_em,
        majority_pm,
        true,
        true,
        SA[false, false, false],
        false,
    )
    model = UnremovableABM(
        typeof(example_agent); properties, scheduler=Schedulers.Randomly()
    )
    for n in 1:n_agents
        if n < n_agents * proportion_incumbents_majority
            policy = rand(SArray{Tuple{2,2,2},Float64,3,8}) * (utilities[1] - utilities[3]) # Scale by utility range
            abm_agent = LearningAgent(
                n,
                policy,
                majority_em,
                majority_pm,
                false,
                true,
                SA[false, false, false],
                false,
            )
        else
            policy = rand(SArray{Tuple{2,2,2},Float64,3,8}) * (utilities[2] - utilities[4]) # Scale by utility range
            abm_agent = LearningAgent(
                n,
                policy,
                minority_em,
                minority_pm,
                true,
                true,
                SA[false, false, false],
                false,
            )
        end
        add_agent!(abm_agent, model)
    end
    return model
end

# function (la::LearningAgent)(info::SVector)
#     la.exploration_probability > rand() && return rand(Bool) # Explore
#     prob_coop = (Fix1(execution_oopsie, la.ε) ∘ Fix1(evaluate, la.policy) ∘ Fix1(mistake, la.α))(info)
#     return prob_coop > rand()
# end
# (la::LearningAgent)(info...) = la(SA[info...])

# Commenting this method out because it's no faster than the above due to a
# custom lerp implementation for 2×2 SArrays:

# function evaluate(policy::P, info::SVector{2, Bool}) where {T,P<:SArray{Tuple{2,2,2},T}}
#     strategy = SArray{NTuple{2,2},T,2,4}(reduce(whichmax, policy; dims=3))
#     return strategy[info[1]+1, info[2]+1]
# end

# TODO: add rng as argument?

# Precompilation
using PrecompileTools

@compile_workload begin
    abm = initialise_abm(;
        n_agents=50,
        norm=iNorm(195),
        majority_pm=SA[0.0, 0.0],
        majority_em=0.01,
        minority_pm=SA[0.0, 0.0],
        minority_em=0.01,
        judge_pm=SA[0.0, 0.0, 0.0],
        judge_em=0.01,
        utilities=SA[2.0, 2.0, 1.0, 1.0],
        learning_rate=0.01,
        exploration_rate=0.01,
        proportion_incumbents_majority=0.9,
    )
    for step in 1:10
        donor = abm.agents[rand(1:nagents(abm))]
        agent_step!(donor, abm)
    end
end

end # module
