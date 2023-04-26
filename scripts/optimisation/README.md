# Optimising IR systems

### What is an IR (Indirect Reciprocity) system?
Agents in an infinite population interact with each other, their actions are judged by an external judge whose strategy is referred to as a "social norm".

The goal of this project is to see which IR system, defined by the agents' strategies and the social norm, lead to fair cooperation.

Agents' strategies (represented as `SMatrix{Tuple{2,2},Bool}`) allow them to base their action on their opponent's (binary) reputation and (binary) "t-shirt colour" (some arbitrary label which I refer to as Red or Blue).

The average reputation of a member of the red or blue groups evolves over time. The reputation dynamics have an explicit solution which is the output of `stationary_incumbent_reputations`.

### What does the optimisation problem look like?

Mathematically:

$\max_x f(x, h(x))$

subject to

$g(x, h(x)) \geq 0$

and

$lb \leq x \leq ub$

where

$x \in \mathbb{R}^n$,

$h : \mathbb{R}^n \rightarrow \mathbb{R}^k$,

$f : \mathbb{R}^n \times \mathbb{R}^k \rightarrow \mathbb{R}^m$,

$g : \mathbb{R}^n \times \mathbb{R}^k \rightarrow \mathbb{R}^p$

Define $\mathbb{U}$ to be the interval $[0, 1]$, then $x \in \mathbb{R}^15$ is made up of:
- Rate of mistakes of agents:
    - Perception mistakes:
        - $\alpha_J \in \mathbb{U}^3$ (Judge)
        - $\alpha_R \in \mathbb{U}^2$ (Red agents)
        - $\alpha_B \in \mathbb{U}^2$ (Blue agents)
    - Execution mistakes:
        - $\varepsilon_i \in \mathbb{U}$ for $i$ one of J, R, B as above
- Utilities of each group $\in \mathbb{R}^4$:
    - Benefit of receiving cooperation for group $i$ (Red or Blue)
    - Cost of cooperating for group $i$ (Red or Blue)
    - `utilities = (b_R, b_B, c_R, c_B)`
- The size of the majority group (Red): `prop_red` or `pR` $\in \mathbb{U}$

$h(x)$ is the average reputation of an agent in each group

$f(x, h(x))$ is the average of the payoffs of each group (unweighted by group size).

$g(x, h(x))$ is the difference between the incumbent payoffs and the payoffs of the $j^{\text{th}}$ mutant.

The problem is feasible if no mutants can invade the incumbents i.e. $g(x, h(x)) \geq 0$.

### How is the code structured?

In `IR`, we define the `Agent` type (`Agent.jl`), and how it is evaluated (with functions from `nlinear_interpolation.jl`).
Then in `payoffs.jl` and `reputation.jl` we provide the functions to "solve" the IR system.
A solution is completed by determining whether any mutants can invade.
There are a total of 16 strategies and hence 16 mutants in each group/population, as a mutant derived from a population is just an agent from that population with a new strategy.

In `scripts/optimisation` there are 6 attempts at implementing the mathematical model described above in Julia
- Four use `JuMP.jl`
- One uses `Optimization.jl`
- One uses `NLPModels.jl` and the unregistered `ObjConsNLPModels.jl`

`jump_attempt_0` uses the previous implementation of `IR` called `TinyIR`, and the rest use `IR`.