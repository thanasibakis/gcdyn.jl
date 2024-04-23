# Methods involving `BranchingProcess` objects.

"``\\frac{1}{1 + exp(-x)}``"
expit(x) = 1 / (1 + exp(-x))

"``\\frac{\\text{yscale}}{1 + exp(-(\\text{xscale} * (x - \text{{xshift}})))} + \\text{yshift}``"
sigmoid(x, xscale, xshift, yscale, yshift) = yscale * expit(xscale * (x - xshift)) + yshift

"""
```julia
type_space_index(model::AbstractBranchingProcess, type)
```

Returns the index of `type` in the given `model`'s type space, or `nothing` if not found.
"""
@memoize function type_space_index(model::AbstractBranchingProcess, type)
    return findfirst(==(type), model.type_space)
end

"""
```julia
λ(model::AbstractBranchingProcess, type)
```

Evaluates the `model`'s birth rate function at the given `type`.
"""
function λ(model::ConstantBranchingProcess, type)
    return model.λ
end

function λ(model::DiscreteBranchingProcess, type)
    i = type_space_index(model, type)

    if isnothing(i)
        throw(ArgumentError("The type must be in the type space of the model."))
    end

    return model.λ[i]
end

function λ(model::SigmoidalBranchingProcess, type)
    return sigmoid(type, model.λ_xscale, model.λ_xshift, model.λ_yscale, model.λ_yshift)
end

"""
```julia
μ(model::AbstractBranchingProcess, type)
```

Evaluates the `model`'s death rate function at the given `type`.
"""
function μ(model::AbstractBranchingProcess, type)
    if isnothing(type_space_index(model, type))
        throw(ArgumentError("The type must be in the type space of the model."))
    end
    
    return model.μ
end

"""
```julia
γ(model::AbstractBranchingProcess, type)
```

Evaluates the `model`'s type change rate function at the given `type` (the rate of change out of `type`).
"""
function γ(model::AbstractBranchingProcess, type)
    i = type_space_index(model, type)

    if isnothing(i)
        throw(ArgumentError("The type must be in the type space of the model."))
    end

    return model.δ * -model.Γ[i, i]
end

"""
```julia
γ(model::AbstractBranchingProcess, from_type, to_type)
```

Evaluates the `model`'s rate of type change from `from_type` to `to_type`.
"""
function γ(model::AbstractBranchingProcess, from_type, to_type)
    if from_type == to_type
        return 0
    end

    i = type_space_index(model, from_type)
    j = type_space_index(model, to_type)

    if isnothing(i) || isnothing(j)
        throw(ArgumentError("The types must be in the type space of the model."))
    end

    return model.δ * model.Γ[i, j]
end

"""
```julia
loglikelihood(model::AbstractBranchingProcess, tree::TreeNode; kw...)
loglikelihood(model::AbstractBranchingProcess, trees::Vector{TreeNode}; kw...)
```

A slight deviation from StatsAPI, as observations are not stored in the model object, so they must be passed as an argument.

Keyword arguments `reltol` and `abstol` are passed to the ODE solver.
"""
function StatsAPI.loglikelihood(
    model::AbstractBranchingProcess{T},
    tree::TreeNode;
    reltol = 1e-3,
    abstol = 1e-3,
    maxiters = 1e5
) where T
    # We may be using autodiff, so we need to know what type T the likelikihood value will be
    p_start = Dict{TreeNode, Vector{T}}()
    p_end = Dict{TreeNode, Vector{T}}()
    logq_start = Dict{TreeNode, T}()
    logq_end = Dict{TreeNode, T}()

    for leaf in LeafTraversal(tree)
        # If σ>0, leaves may be at non-present times, so it's incorrect to initialize
        # p to be 1-ρ for all leaf times
        if leaf.time == model.present_time
            p_end[leaf] = fill(1 - model.ρ, size(model.type_space))
        end

        # Be sure to specify the iip=true of the ODEProblem for type stability
        # and fewer memory allocations
        p = solve(
            ODEProblem{true}(
                dp_dt!,
                fill(1 - model.ρ, size(model.type_space)),
                (0, model.present_time - leaf.time),
                model
            ),
            Tsit5();
            isoutofdomain = (p, args, t) -> any(x -> x < 0 || x > 1, p),
            save_everystep = false,
            save_start = false,
            reltol = reltol,
            abstol = abstol,
            maxiters = maxiters
        )

        if !SciMLBase.successful_retcode(p)
            @warn "Leaf initial condition could not be solved. Exit code $(p.retcode). The integration timespan was $(model.present_time - leaf.time). Density will evaluate to zero."

            return -Inf
        end

        p_end[leaf] = p.u[end]
    end

    for event in PostOrderTraversal(tree.children[1])
        t_start = model.present_time - event.up.time
        t_end = model.present_time - event.time

        if event.event == :sampled_survival
            # event already has p_end
            logq_end[event] = log(model.ρ)
        elseif event.event == :sampled_death
            # event already has p_end
            logq_end[event] = log(μ(model, event.up.type)) + log(model.σ)
        elseif event.event == :birth
            p_end[event] = p_start[event.children[1]]
            logq_end[event] = (
                log(λ(model, event.up.type))
                + logq_start[event.children[1]]
                + logq_start[event.children[2]]
            )
        
        elseif event.event == :type_change
            if event.type == event.up.type
                @warn "Self-loop encountered at a type change event in the tree. Density will evaluate to zero."
                return -Inf
            end

            p_end[event] = p_start[event.children[1]]
            logq_end[event] = (
                log(γ(model, event.up.type, event.type))
                + logq_start[event.children[1]]
            )
        else
            throw(ArgumentError("Unknown event type $(event.event)"))
        end

        p_logq = solve(
            ODEProblem{true}(
                dp_logq_dt!,
                [p_end[event]; logq_end[event]],
                (t_end, t_start),
                (model, event.up.type)
            ),
            Tsit5();
            isoutofdomain = (p_logq, args, t) -> any(x -> x < 0 || x > 1, p_logq[1:end-1]),
            save_everystep = false,
            save_start = false,
            reltol = reltol,
            abstol = abstol
        )

        if !SciMLBase.successful_retcode(p_logq)
            @warn "Density of branch could not be solved. Exit code $(p_logq.retcode). The integration timespan was $(t_start - t_end). Density of tree will evaluate to zero."
            return -Inf
        end

        p_start[event] = p_logq.u[end][1:end-1]
        logq_start[event] = p_logq.u[end][end]
    end

    result = logq_start[tree.children[1]]

    # Compute the non-observation probability of the whole tree
    p = solve(
        ODEProblem{true}(
            dp_dt!,
            fill(1 - model.ρ, size(model.type_space)),
            (0, model.present_time),
            model
        ),
        Tsit5();
        isoutofdomain = (p, args, t) -> any(x -> x < 0 || x > 1, p),
        save_everystep = false,
        save_start = false,
        reltol = reltol,
        abstol = abstol
    )

    if !SciMLBase.successful_retcode(p)
        @warn "Non-observation probability of tree could not be solved. Exit code $(p.retcode). The integration timespan was $(model.present_time). Density will evaluate to zero."
        return -Inf
    end

    # Condition on observation of at least one lineage
    p_i = p.u[end][type_space_index(model, tree.type)]
    result -= log(1 - p_i)

    return result
end

function StatsAPI.loglikelihood(
    model::AbstractBranchingProcess{T},
    trees::Vector{TreeNode};
    reltol = 1e-3,
    abstol = 1e-3
) where T
    return sum(StatsAPI.loglikelihood(model, tree; reltol=reltol, abstol=abstol) for tree in trees)
end

"""
See equation (1) of this paper:

Barido-Sottani, Joëlle, Timothy G Vaughan, and Tanja Stadler. “A Multitype Birth–Death Model for Bayesian Inference of Lineage-Specific Birth and Death Rates.” Edited by Adrian Paterson. Systematic Biology 69, no. 5 (September 1, 2020): 973–86. https://doi.org/10.1093/sysbio/syaa016.
"""
function dp_dt!(dp, p, model::AbstractBranchingProcess, t)
    for (i, type) in enumerate(model.type_space)
        λₓ = λ(model, type)
        μₓ = μ(model, type)

        dp[i] = (
            -(λₓ + μₓ) * p[i]
            + μₓ * (1 - model.σ)
            + λₓ * p[i]^2
            + model.δ * sum(model.Γ[i, j] * p[j] for j in 1:length(model.type_space))
        )
    end
end

"""
See equations (1) and (2) of this paper:

Barido-Sottani, Joëlle, Timothy G Vaughan, and Tanja Stadler. “A Multitype Birth–Death Model for Bayesian Inference of Lineage-Specific Birth and Death Rates.” Edited by Adrian Paterson. Systematic Biology 69, no. 5 (September 1, 2020): 973–86. https://doi.org/10.1093/sysbio/syaa016.
"""
function dp_logq_dt!(dp_logq, p_logq, args, t)
    p = view(p_logq, 1:lastindex(p_logq)-1)
    model, parent_type = args

    λₓ = λ(model, parent_type)
    μₓ = μ(model, parent_type)
    γₓ = γ(model, parent_type)

    p_i = p[type_space_index(model, parent_type)]
    dlogq_i = -(λₓ + μₓ + γₓ) + 2 * λₓ * p_i

    # Need to pass a view instead of a slice, to pass by reference instead of value
    dp = view(dp_logq, 1:lastindex(dp_logq)-1)
    dp_dt!(dp, p, model, t)
    dp_logq[end] = dlogq_i
end

"""
```julia
rand_tree(model, [n,] init_type; reject_stubs=true)
```

Generate `n` random trees from the given model (optional parameter, default is `1`), each starting at the given initial type.

Note that trees will have root time `0`, with time increasing toward the tips.
This aligns us with the branching process models, but contradicts the notion of time typically used in phylogenetics.

Can optionally choose not to `reject_stubs`, equivalent to not conditioning on the root node having at least one sampled descendant.

See also [`TreeNode`](@ref), [`sample_child!`](@ref), [`mutate!`](@ref).
"""
function rand_tree(
    model::AbstractBranchingProcess,
    n,
    init_type;
    reject_stubs=true
)
    trees = Vector{TreeNode}(undef, n)

    for i in 1:n
        trees[i] = rand_tree(model, init_type; reject_stubs=reject_stubs)
    end

    return trees
end

function rand_tree(
    model::AbstractBranchingProcess,
    init_type;
    reject_stubs=true
)
    root = TreeNode(:root, 0, init_type)
    needs_children = [root]

    # Evolve a fully-observed tree
    while length(needs_children) > 0
        node = pop!(needs_children)
        child = sample_child!(node, model)

        if child.event == :birth
            push!(needs_children, child)
            push!(needs_children, child)
        elseif child.event == :type_change
            push!(needs_children, child)
        end
    end

    # Flag every node that has a sampled descendant
    has_sampled_descendant = Set{TreeNode}()

    for leaf in Leaves(root)
        if leaf.event ∉ (:sampled_survival, :sampled_death)
            continue
        end

        node = leaf

        while !isnothing(node)
            push!(has_sampled_descendant, node)
            node = node.up
        end
    end

    # Prune subtrees that don't have any sampled descendants
    for node in PostOrderDFS(root)
        if node.event ∈ (:birth, :root)
            filter!(in(has_sampled_descendant), node.children)

            if node.event == :birth && length(node.children) == 1
                delete!(node)
            end
        end
    end

    # If we're rejecting stubs and have one, try again
    if reject_stubs && length(root.children) == 0
        return rand_tree(model, init_type; reject_stubs = true)
    end

    return root
end

"""
```julia
mutate!(node, model)
```

Change the type of the given node according to the transition probabilities or rates in the given model.
"""
function mutate!(node::TreeNode, model::AbstractBranchingProcess)
    if node.type ∉ model.type_space
        throw(ArgumentError("The type of the node must be in the type space of the mutator."))
    end

    i = type_space_index(model, node.type)
    transition_probs = model.Γ[i, :] ./ -model.Γ[i, i]
    transition_probs[i] = 0
    node.type = sample(model.type_space, Weights(transition_probs))
end

"""
```julia
sample_child!(parent, model)
```

Append a new child event to the given parent node, selecting between events from [`EVENTS`](@ref) according to the given model's rate parameters.

Note that the child will a time `t` larger than its parent.
This aligns us with the branching process models, but contradicts the notion of time typically used in phylogenetics.
"""
function sample_child!(parent::TreeNode, model::AbstractBranchingProcess)
    if length(parent.children) == 2
        throw(ArgumentError("Can only have 2 children max"))
    end

    λₓ, μₓ, γₓ = λ(model, parent.type), μ(model, parent.type), γ(model, parent.type)

    waiting_time = rand(Exponential(1 / (λₓ + μₓ + γₓ)))

    if parent.time + waiting_time > model.present_time
        event = sample([:sampled_survival, :unsampled_survival], Weights([model.ρ, 1 - model.ρ]))
        child = TreeNode(event, model.present_time, parent.type)
    else
        event = sample([:birth, :unsampled_death, :type_change], Weights([λₓ, μₓ, γₓ]))

        if event == :unsampled_death && rand() < model.σ
            event = :sampled_death
        end

        child = TreeNode(event, parent.time + waiting_time, parent.type)

        if event == :type_change
            mutate!(child, model)
        end
    end

    attach!(parent, child)

    return child
end

"""
```julia
map_types(tree, mapping; prune_self_loops = true)
```

Replaces the type attribute of all nodes in `tree` with the result of the callable `mapping` applied to the type values.

Optionally but by default, if the map results in a type change event with the same type as its parent,
the type change event is pruned from the tree.
"""
function map_types!(mapping, tree; prune_self_loops = true)
    for node in PreOrderTraversal(tree)
        node.type = mapping(node.type)

        # If a type change resulted in an type of the same bin as the parent,
        # that isn't a valid type change in the CTMC, so we prune it
        if prune_self_loops && node.event == :type_change && node.type == node.up.type
            delete!(node)
        end
    end
end
