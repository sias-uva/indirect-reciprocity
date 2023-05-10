# Indirect Reciprocity (IR)

This repository contains the source code for my work on the (algorithmic) game theory topic of indirect reciprocity: a mechanism for encouraging cooperation between self-interested agents.
My work focuses on the interplay between cooperation and fairness, investigating how we can engineer systems of indirect reciprocity, whether artificial or otherwise, to achieve fair, socially desirable outcomes.

### What is indirect reciprocity?

Indirect reciprocity allows agents to observe (directly or indirectly) and judge the actions of others, then use these judgements in determining whether to cooperate or defect with an individual in the future.
This allows agents who have never interacted before to make better-informed decisions taking into account the other agent's reputation.

What we observe in real life is that reputation mechanisms are presented with other information that agents use when making decisions.
For example, someone's name on a job application may be used to qualify an applicant's credentials, or, on a site such as Airbnb, a host may discriminate based on race.

This work combines the mechanisms of homophily (or more broadly, tag-based cooperation) and indirect reciprocity to see how fair cooperation can be achieved, and the barriers to its realisation that may be present in social systems.

### Structure of the repository
- `IR` is a Julia module/package containing all of the core source code to determine the reputations and payoffs in an evolutionary game theory model of tag-based and reputation-based cooperation. As the science machine is cranked, this will be made into a stand-alone repository.
- `scripts` uses the local `IR` package to do the science including exploration, plots, and a whole lot of errors.
