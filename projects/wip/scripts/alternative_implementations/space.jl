using IR: IR
using IR: lerp, mistake
using GLMakie
using Agents
using StaticArrays
using Base: Fix1

whichmax(x, y) = argmax((x, y)) - 1

Judge{B,C} = IR.Agent{2,Bool,B,C,SArray{NTuple{2,2},Bool}}

mutable struct LearningAgent{P,E,A} <: AbstractAgent
    id::Int
    pos::NTuple{2,Int}
    policy::P
    ε::E
    α::A
    reputation::Bool
    vision_range::Int64
    memory::SVector{2,Bool}
    has_interacted_as_donor::Bool
end

function evaluate(policy::P, info::Real) where {T,P<:SArray{Tuple{2,2},T}}
    strategy = SVector{2,T}(reduce(whichmax, policy; dims=2))
    return lerp(strategy, info)
end

function cooperation_probability(donor, recipient)
    return (Fix1(mistake, donor.ε) ∘ Fix1(evaluate, donor.policy) ∘ Fix1(mistake, donor.α))(
        recipient.reputation
    )
end

"""
    find_optimal_position(agent, space)

Calculate the grid position that gives the highest expected return given that
the agent plays its current strategy. 
"""
function find_optimal_position(agent, model)
    # TODO: optimise by calculating value for each agent in vision + 1 (position
    # + neighbouring agents) then the value of the cell is the summed value of
    # its neighbours
    argmax(nearby_positions(agent, model, agent.vision_range)) do position
        get = 0.0
        give = 0.0
        # For each neighbouring agent,
        agents_nearby_position = nearby_agents(position, model, 1)
        for other_agent in agents_nearby_position
            # how much will I give them...
            give +=
                model.cost * cooperation_probability(agent, other_agent) /
                length(collect(agents_nearby_position))
            # and how much will I get from them?
            get +=
                model.benefit * cooperation_probability(other_agent, agent) /
                length(collect(nearby_agents(other_agent, model, 1)))
        end
        get - give
    end
end

function move_towards_optimal!(agent, model)
    destination = find_optimal_position(agent, model)
    plan_route!(agent, destination, model.properties.pathfinder)
    return move_along_route!(agent, model, model.properties.pathfinder)
end

function initialise_abm(;
    n_agents=5,
    grid_size=(10, 10),
    norm=12,
    player_pm=0.0,
    player_em=0.01,
    judge_pm=SA[0.0, 0.0],
    judge_em=0.01,
    utilities=SA[2.0, 1.0],
    learning_rate=0.01,
    exploration_rate=0.01,
    vision_range=100,
)
    _norm = IR.iStrategy(norm)
    judge = IR.Agent(_norm, judge_em, judge_pm)
    space = GridSpace(grid_size; metric=:manhattan)
    pathfinder = Agents.Pathfinding.AStar(space; diagonal_movement=false)
    properties = (;
        judge,
        benefit=utilities[1],
        cost=utilities[2],
        learning_rate,
        exploration_rate,
        pathfinder,
    )
    example_strategy = rand(SArray{Tuple{2,2},Float64,2,4})
    example_agent = LearningAgent(
        -1,
        NTuple{2,Int}(rand(Int, 2)),
        example_strategy,
        player_em,
        player_pm,
        true,
        vision_range,
        SA[false, false],
        false,
    )
    model = UnremovableABM(
        typeof(example_agent), space; properties, scheduler=Schedulers.Randomly()
    )
    for n in 1:n_agents
        policy = rand(SArray{Tuple{2,2},Float64,2,4}) * (utilities[1] - utilities[2]) # Scale by utility range
        abm_agent = LearningAgent(
            n,
            NTuple{2,Int}(rand(Int, 2)),
            policy,
            player_em,
            player_pm,
            true,
            vision_range,
            SA[false, false],
            false,
        )
        add_agent!(abm_agent, model)
    end
    return model
end

function agent_step!(donor, model)
    # Move then interact with a random nearby agent
    if rand() > 0.05
        move_towards_optimal!(donor, model)
    else
        move_agent!(donor, random_nearby_position(donor.pos, model), model)
    end
    # Interact
    recipient = random_nearby_agent(donor, model, 1)
    isnothing(recipient) || play_and_learn!((donor, recipient), model)
    return nothing
end

function play_and_learn!((donor, recipient)::NTuple{2,LearningAgent}, model)
    donor.has_interacted_as_donor = true
    info = recipient.reputation
    perceived_info = mistake(donor.α, info) > rand()
    # println("Perceived: $perceived_info, index: $(perceived_info .+ 1)")
    if model.exploration_rate > rand()
        outcome = rand(Bool) # Explore
    else
        prob_coop = (Fix1(mistake, donor.ε) ∘ Fix1(evaluate, donor.policy))(perceived_info)
        outcome = prob_coop > rand() # Q-learning
    end
    # outcome = softmax(donor.policy[perceived_info .+ 1..., :])[1] > rand() # Roth-Erev learning
    judgement = model.judge(SA[info, outcome]) > rand()
    donor.reputation = judgement
    donor.memory = SA[perceived_info, outcome]
    # Agents learn whether or not there was a donation, utilities adjusted
    # accordingly. No donation => no cost, no benefit but decay of corresponding
    # Q-value still takes place.
    # println("Learning")
    # println("Donor $(donor.id) donated to $(recipient.id)? ", outcome)
    learn!(donor, outcome * -model.cost, model)
    learn!(recipient, outcome * model.benefit, model)
    return nothing
end

function learn!(la::LearningAgent, utility, model)
    !la.has_interacted_as_donor && return nothing
    interaction = la.memory
    # println("Agent $(la.id) learned $utility from $(la.memory)")
    idx = interaction .+ 1
    ϕ = model.learning_rate
    old_policy = la.policy
    new_q = (1 - ϕ) * old_policy[idx...] + ϕ * utility # Q-learning
    linear_idx = interaction[1] + 2 * interaction[2] + 1
    la.policy = setindex(old_policy, new_q, linear_idx)
    return nothing
end

# heatarray = :temperature
# heatkwargs = (colorrange = (-20, 60), colormap = :thermal)
# plotkwargs = (;
#     ac = daisycolor, as, am,
#     scatterkwargs = (strokewidth = 1.0,),
#     heatarray, heatkwargs
# )

model = initialise_abm(;
    n_agents=100, grid_size=(40, 40), exploration_rate=0.05, learning_rate=0.001
)

abmvideo(
    "figures/web/anim_man.mp4",
    model,
    agent_step!;
    ac=ac(a) = a.reputation == false ? "#2b2b33" : "#bf2642",
    frames=10,
    framerate=60,
)

# abmplot(model; agent_step!,
#     ac = ac(a) = a.reputation == false ? "#2b2b33" : "#bf2642",
# ) 
