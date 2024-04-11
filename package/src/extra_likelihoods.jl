# Alternate likelihoods for multitype branching processes, with various assumptions.

"""
```julia
naive_loglikelihood(model, tree)
```

Compute the log-likelihood of the given model under the assumption that the given tree is fully observed (ie. all nodes are sampled, dead or not).
"""
function naive_loglikelihood(model::BranchingProcess, tree::TreeNode)
    result = 0.0

    for node in PostOrderTraversal(tree.children[1])
        λₓ, μₓ, γₓ = λ(model, node.up.type), μ(model, node.up.type), γ(model, node.up.type)
        Λₓ = λₓ + μₓ + γₓ

        if node.event == :birth
            result += logpdf(Exponential(1 / Λₓ), node.time - node.up.time) + log(λₓ / Λₓ)
        elseif node.event == :sampled_death
            result += logpdf(Exponential(1 / Λₓ), node.time - node.up.time) + log(μₓ / Λₓ)
        elseif node.event == :type_change
            transition_rate = γ(model, node.up.type, node.type)
            result += logpdf(Exponential(1 / Λₓ), node.time - node.up.time) + log(transition_rate) - log(Λₓ)
        elseif node.event == :sampled_survival
            result += log(1 - cdf(Exponential(1 / Λₓ), node.time - node.up.time)) + log(model.ρ)
        elseif node.event == :unsampled_survival
            result += log(1 - cdf(Exponential(1 / Λₓ), node.time - node.up.time)) + log(1 - model.ρ)
        else
            throw(ArgumentError("Tree contains incompatible event $(node.event)"))
        end
    end

    return result
end

"""
```julia
stadler_appx_loglikelhood(model, tree)
```

Compute the log-likelihood of the given model under the assumption that all type change nodes have at least one sampled descendant.

Barido-Sottani, Joëlle, Timothy G Vaughan, and Tanja Stadler. “A Multitype Birth–Death Model for Bayesian Inference of Lineage-Specific Birth and Death Rates.” Edited by Adrian Paterson. Systematic Biology 69, no. 5 (September 1, 2020): 973–86. https://doi.org/10.1093/sysbio/syaa016.
"""
function stadler_appx_loglikelihood(model::BranchingProcess, tree::TreeNode)
    result = 0
    ρ, σ, present_time = model.ρ, model.σ, model.present_time

    for node in PostOrderTraversal(tree.children[1])
        λₓ, μₓ, γₓ = λ(model, node.up.type), μ(model, node.up.type), γ(model, node.up.type)
        Λₓ = λₓ + μₓ + γₓ
        c = √(Λₓ^2 - 4 * μₓ * (1 - σ) * λₓ)
        x = (-Λₓ - c) / 2
        y = (-Λₓ + c) / 2

        helper(t) = (y + λₓ * (1 - ρ)) * exp(-c * t) - x - λₓ * (1 - ρ)

        t_s = present_time - node.up.time
        t_e = present_time - node.time

        log_f_N = c * (t_e - t_s) + 2 * (log(helper(t_e)) - log(helper(t_s)))

        result += log_f_N

        if node.event == :birth
            result += log(λₓ)
        elseif node.event == :sampled_death
            result += log(σ) + log(μₓ)
        elseif node.event == :type_change
            transition_rate = γ(model, node.up.type, node.type)
            result += log(transition_rate)
        elseif node.event == :sampled_survival
            result += log(ρ)
        else
            throw(ArgumentError("Tree contains incompatible event $(node.event)"))
        end
    end

    # Condition on tree not getting rejected as a stub
    # Let p be the extinction (or more generally, emptiness) prob
    
    p = let
        λₓ, μₓ, γₓ = λ(model, tree.type), μ(model, tree.type), γ(model, tree.type)

        Λₓ = λₓ + μₓ + γₓ
        root_time = present_time - 0
        c = √(Λₓ^2 - 4 * μₓ * (1 - σ) * λₓ)
        x = (-Λₓ - c) / 2
        y = (-Λₓ + c) / 2

        (
            -1
            / λₓ
            * ((y + λₓ * (1 - ρ)) * x * exp(-c * root_time) - y * (x + λₓ * (1 - ρ)))
            / ((y + λₓ * (1 - ρ)) * exp(-c * root_time) - (x + λₓ * (1 - ρ)))
        )
    end

    result -= log(1 - p)

    return result
end

"""
```julia
stadler_appx_unconditioned_loglikelhood(model, tree)
```

Compute the log-likelihood of the given model under the assumption that all type change nodes have at least one sampled descendant.

Does not condition on all trees having at least one sampled descendant (ie. [`rand_tree`](@ref) has keyword argument `reject_stubs=False`).

Barido-Sottani, Joëlle, Timothy G Vaughan, and Tanja Stadler. “A Multitype Birth–Death Model for Bayesian Inference of Lineage-Specific Birth and Death Rates.” Edited by Adrian Paterson. Systematic Biology 69, no. 5 (September 1, 2020): 973–86. https://doi.org/10.1093/sysbio/syaa016.
"""
function stadler_appx_unconditioned_loglikelihood(model::BranchingProcess, tree::TreeNode)
    result = 0
    ρ, σ, present_time = model.ρ, model.σ, model.present_time

    for node in PostOrderTraversal(tree.children[1])
        λₓ, μₓ, γₓ = λ(model, node.up.type), μ(model, node.up.type), γ(model, node.up.type)

        Λₓ = λₓ + μₓ + γₓ
        c = √(Λₓ^2 - 4 * μₓ * (1 - σ) * λₓ)
        x = (-Λₓ - c) / 2
        y = (-Λₓ + c) / 2

        helper(t) = (y + λₓ * (1 - ρ)) * exp(-c * t) - x - λₓ * (1 - ρ)

        t_s = present_time - node.up.time
        t_e = present_time - node.time

        log_f_N = c * (t_e - t_s) + 2 * (log(helper(t_e)) - log(helper(t_s)))

        result += log_f_N

        if node.event == :birth
            result += log(λₓ)
        elseif node.event == :sampled_death
            result += log(σ) + log(μₓ)
        elseif node.event == :type_change
            transition_rate = γ(model, node.up.type, node.type)
            result += log(transition_rate)
        elseif node.event == :sampled_survival
            result += log(ρ)
        else
            throw(ArgumentError("Tree contains incompatible event $(node.event)"))
        end
    end

    return result
end
