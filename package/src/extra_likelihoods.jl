# Alternate likelihoods for multitype branching processes, with various assumptions.

"""
```julia
naive_loglikelihood(model, tree)
```

Compute the log-likelihood of the given model under the assumption that the given tree is fully observed (ie. all nodes are sampled, dead or not).
"""
function naive_loglikelihood(model::AbstractBranchingProcess, tree::TreeNode)
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
function stadler_appx_loglikelihood(model::AbstractBranchingProcess, tree::TreeNode)
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
function stadler_appx_unconditioned_loglikelihood(model::AbstractBranchingProcess, tree::TreeNode)
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

"""
```julia
loglikelihood(model::AbstractBranchingProcess, tree::TreeNode; kw...)
```

Equivalent to our [`StatsAPI.loglikelihood`](@ref), but operates with ODEs on the natural scale, not log scale.
"""
function natural_ode_loglikelihood(
    model::AbstractBranchingProcess,
    tree::TreeNode;
    reltol = 1e-3,
    abstol = 1e-3
)

    ρ, σ = model.ρ, model.σ
    present_time = model.present_time

    # We may be using autodiff, so figure out what type the likelikihood value will be
    T = typeof(λ(model, model.type_space[1]))

    p_start = Dict{TreeNode, Vector{T}}()
    p_end = Dict{TreeNode, Vector{T}}()
    q_start = Dict{TreeNode, T}()
    q_end = Dict{TreeNode, T}()

    # If σ>0, leaves may be at non-present times, so it's incorrect to initialize
    # p to be 1-ρ for all leaf times
    for leaf in LeafTraversal(tree)
        # Be sure to specify the iip=true of the ODEProblem for type stability
        # and fewer memory allocations
        p = solve(
            ODEProblem{true}(
                dp_dt!,
                ones(Float64, axes(model.type_space)) .- ρ,
                (0, present_time - leaf.time),
                model
            ),
            Tsit5();
            isoutofdomain = (p, args, t) -> any(x -> x < 0 || x > 1, p),
            save_everystep = false,
            save_start = false,
            reltol = reltol,
            abstol = abstol
        )

        p_end[leaf] = p.u[end]
    end

    for event in PostOrderTraversal(tree.children[1])
        t_start = present_time - event.up.time
        t_end = present_time - event.time

        if event.event == :sampled_survival
            # event already has p_end
            q_end[event] = ρ
        elseif event.event == :sampled_death
            # event already has p_end
            q_end[event] = μ(model, event.up.type) * σ
        elseif event.event == :birth
            p_end[event] = p_start[event.children[1]]
            q_end[event] = (
                λ(model, event.up.type) * q_start[event.children[1]] * q_start[event.children[2]]
            )
        elseif event.event == :type_change
            if event.type == event.up.type
                @warn "Self-loop encountered at a type change event in the tree. Density will evaluate to zero."
                return -Inf
            end

            p_end[event] = p_start[event.children[1]]
            q_end[event] = (
                γ(model, event.up.type, event.type)
                * q_start[event.children[1]]
            )
        else
            throw(ArgumentError("Unknown event type $(event.event)"))
        end

        pq = solve(
            ODEProblem{true}(
                dpq_dt!,
                [p_end[event]; q_end[event]],
                (t_end, t_start),
                (model, event.up.type)
            ),
            Tsit5();
            isoutofdomain = (pq, args, t) -> any(x -> x < 0, pq),
            save_everystep = false,
            save_start = false,
            reltol = reltol,
            abstol = abstol
        )

        p_start[event] = pq.u[end][1:end-1]
        q_start[event] = pq.u[end][end]
    end

    result = log(q_start[tree.children[1]])

    # Non-extinction probability

    p = solve(
        ODEProblem{true}(
            dp_dt!,
            ones(Float64, axes(model.type_space)) .- ρ,
            (0, present_time),
            model
        ),
        Tsit5();
        isoutofdomain = (p, args, t) -> any(x -> x < 0 || x > 1, p),
        save_everystep = false,
        save_start = false,
        reltol = reltol,
        abstol = abstol
    )

    p_i = p.u[end][type_space_index(model, tree.type)]
    result -= log(1 - p_i)

    return result
end

"""
See equations (1) and (2) of this paper:

Barido-Sottani, Joëlle, Timothy G Vaughan, and Tanja Stadler. “A Multitype Birth–Death Model for Bayesian Inference of Lineage-Specific Birth and Death Rates.” Edited by Adrian Paterson. Systematic Biology 69, no. 5 (September 1, 2020): 973–86. https://doi.org/10.1093/sysbio/syaa016.
"""
function dpq_dt!(dpq, pq, args, t)
    p, q_i = view(pq, 1:lastindex(pq)-1), pq[end]
    model, parent_type = args

    λₓ = λ(model, parent_type)
    μₓ = μ(model, parent_type)
    γₓ = γ(model, parent_type)

    p_i = p[type_space_index(model, parent_type)]
    dq_i = -(λₓ + μₓ + γₓ) * q_i + 2 * λₓ * q_i * p_i

    # Need to pass a view instead of a slice, to pass by reference instead of value
    dp = view(dpq, 1:lastindex(dpq)-1)
    dp_dt!(dp, p, model, t)
    dpq[end] = dq_i
end

"""
See equation (1) of this paper:

Barido-Sottani, Joëlle, Timothy G Vaughan, and Tanja Stadler. “A Multitype Birth–Death Model for Bayesian Inference of Lineage-Specific Birth and Death Rates.” Edited by Adrian Paterson. Systematic Biology 69, no. 5 (September 1, 2020): 973–86. https://doi.org/10.1093/sysbio/syaa016.
"""
function dp_dt!(dp, p, model::AbstractBranchingProcess, t)
    for (i, type) in enumerate(model.type_space)
        λₓ = λ(model, type)
        μₓ = μ(model, type)
        γₓ = γ(model, type)

        dp[i] = (
            -(λₓ + μₓ + γₓ) * p[i]
            + μₓ * (1 - model.σ)
            + λₓ * p[i]^2
            + sum(γ(model, type, to_type) * p[j] for (j, to_type) in enumerate(model.type_space))
        )
    end
end
