# Implements various alternative densities for a MultitypeBranchingProcess

function naive_loglikelihood(model::MultitypeBranchingProcess, tree::TreeNode)
    result = 0
    ρ = model.ρ
    state_space, transition_matrix = model.state_space, model.transition_matrix

    for node in AbstractTrees.PostOrderDFS(tree.children[1])
        λ, μ, γ = model.λ(node.up.phenotype), model.μ(node.up.phenotype), model.γ(node.up.phenotype)
        Λ = λ + μ + γ

        if node.event == :birth
            result += logpdf(Exponential(1 / Λ), node.t - node.up.t) + log(λ / Λ)
        elseif node.event == :sampled_death
            result += logpdf(Exponential(1 / Λ), node.t - node.up.t) + log(μ / Λ)
        elseif node.event == :mutation
            mutation_prob = transition_matrix[findfirst(state_space .== node.up.phenotype), findfirst(state_space .== node.phenotype)]

            result += logpdf(Exponential(1 / Λ), node.t - node.up.t) + log(γ / Λ) + log(mutation_prob)
        elseif node.event == :sampled_survival
            result += log(1 - cdf(Exponential(1 / Λ), node.t - node.up.t)) + log(ρ)
        elseif node.event == :unsampled_survival
            result += log(1 - cdf(Exponential(1 / Λ), node.t - node.up.t)) + log(1 - ρ)
        else
            throw(ArgumentError("Tree contains incompatible event $(node.event)"))
        end
    end

    return result
end

function stadler_appx_loglikelhood(model::MultitypeBranchingProcess, tree::TreeNode)
    result = 0
    ρ, σ, present_time = model.ρ, model.σ, model.present_time
    state_space, transition_matrix = model.state_space, model.transition_matrix

    for node in AbstractTrees.PostOrderDFS(tree.children[1])
        λ, μ, γ = model.λ(node.up.phenotype), model.μ(node.up.phenotype), model.γ(node.up.phenotype)

        Λ = λ + μ + γ
        c = √(Λ^2 - 4 * μ * (1 - σ) * λ)
        x = (-Λ - c) / 2
        y = (-Λ + c) / 2

        helper(t) = (y + λ * (1 - ρ)) * exp(-c * t) - x - λ * (1 - ρ)

        t_s = present_time - node.up.t
        t_e = present_time - node.t

        log_f_N = c * (t_e - t_s) + 2 * (log(helper(t_e)) - log(helper(t_s)))

        result += log_f_N

        if node.event == :birth
            result += log(λ)
        elseif node.event == :sampled_death
            result += log(σ) + log(μ)
        elseif node.event == :mutation
            mutation_prob = transition_matrix[findfirst(state_space .== node.up.phenotype), findfirst(state_space .== node.phenotype)]

            result += log(γ) + log(mutation_prob)
        elseif node.event == :sampled_survival
            result += log(ρ)
        else
            throw(ArgumentError("Tree contains incompatible event $(node.event)"))
        end
    end

    # Condition on tree not getting rejected as a stub
    # Let p be the extinction (or more generally, emptiness) prob
    λ, μ, γ = model.λ(tree.phenotype), model.μ(tree.phenotype), model.γ(tree.phenotype)

    Λ = λ + μ + γ
    root_time = present_time - 0
    c = √(Λ^2 - 4 * μ * (1 - σ) * λ)
    x = (-Λ - c) / 2
    y = (-Λ + c) / 2

    p = (
        -1
        / λ
        * ((y + λ * (1 - ρ)) * x * exp(-c * root_time) - y * (x + λ * (1 - ρ)))
        / ((y + λ * (1 - ρ)) * exp(-c * root_time) - (x + λ * (1 - ρ)))
    )

    result -= log(1 - p)

    return result
end

function stadler_appx_unconditioned_loglikelhood(model::MultitypeBranchingProcess, tree::TreeNode)
    result = 0
    ρ, σ, present_time = model.ρ, model.σ, model.present_time
    state_space, transition_matrix = model.state_space, model.transition_matrix

    for node in AbstractTrees.PostOrderDFS(tree.children[1])
        λ, μ, γ = model.λ(node.up.phenotype), model.μ(node.up.phenotype), model.γ(node.up.phenotype)

        Λ = λ + μ + γ
        c = √(Λ^2 - 4 * μ * (1 - σ) * λ)
        x = (-Λ - c) / 2
        y = (-Λ + c) / 2

        helper(t) = (y + λ * (1 - ρ)) * exp(-c * t) - x - λ * (1 - ρ)

        t_s = present_time - node.up.t
        t_e = present_time - node.t

        log_f_N = c * (t_e - t_s) + 2 * (log(helper(t_e)) - log(helper(t_s)))

        result += log_f_N

        if node.event == :birth
            result += log(λ)
        elseif node.event == :sampled_death
            result += log(σ) + log(μ)
        elseif node.event == :mutation
            mutation_prob = transition_matrix[findfirst(state_space .== node.up.phenotype), findfirst(state_space .== node.phenotype)]

            result += log(γ) + log(mutation_prob)
        elseif node.event == :sampled_survival
            result += log(ρ)
        else
            throw(ArgumentError("Tree contains incompatible event $(node.event)"))
        end
    end

    return result
end
