# Methods involving `AbstractBranchingProcess` objects.

"``\\frac{1}{1 + exp(-x)}``"
expit(x) = 1 / (1 + exp(-x))

"``\\frac{\\text{yscale}}{1 + exp(-(\\text{xscale} * (x - \text{{xshift}})))} + \\text{yshift}``"
sigmoid(x, xscale, xshift, yscale, yshift) = yscale * expit(xscale * (x - xshift)) + yshift

"""
```julia
λ(model::AbstractBranchingProcess, type)
```

Evaluates the `model`'s birth rate function at the given `type`.
"""
function λ(model::AbstractBranchingProcess, type)
    return sigmoid(type, model.λ_xscale, model.λ_xshift, model.λ_yscale, model.λ_yshift)
end

"""
```julia
μ(model::AbstractBranchingProcess, type)
```

Evaluates the `model`'s death rate function at the given `type`.
"""
function μ(model::AbstractBranchingProcess, type)
    return model.μ
end

"""
```julia
γ(model::AbstractBranchingProcess, type)
```

Evaluates the `model`'s type change rate function at the given `type` (the rate of change out of `type`).
"""
function γ(model::FixedTypeChangeRateBranchingProcess, type)
    return model.γ
end

function γ(model::VaryingTypeChangeRateBranchingProcess, type)
    i = findfirst(==(type), model.type_space)

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
function γ(model::FixedTypeChangeRateBranchingProcess, from_type, to_type)
    if from_type == to_type
        return 0
    end

    i = findfirst(==(from_type), model.type_space)
    j = findfirst(==(to_type), model.type_space)

    if isnothing(i) || isnothing(j)
        throw(ArgumentError("The types must be in the type space of the model."))
    end

    return model.γ * model.Π[i, j]
end

function γ(model::VaryingTypeChangeRateBranchingProcess, from_type, to_type)
    if from_type == to_type
        return 0
    end

    i = findfirst(==(from_type), model.type_space)
    j = findfirst(==(to_type), model.type_space)

    if isnothing(i) || isnothing(j)
        throw(ArgumentError("The types must be in the type space of the model."))
    end

    return model.δ * model.Γ[i, j]
end

# To allow us to broadcast the rate parameter functions over types
Base.broadcastable(model::AbstractBranchingProcess) = Ref(model)

"""
```julia
loglikelihood(model::AbstractBranchingProcess, tree::TreeNode; kw...)
loglikelihood(model::AbstractBranchingProcess, trees; kw...)
```

A slight deviation from StatsAPI, as observations are not stored in the model object, so they must be passed as an argument.

Keyword arguments `reltol` and `abstol` are passed to the ODE solver.
"""
function StatsAPI.loglikelihood(
    model::AbstractBranchingProcess,
    tree::TreeNode;
    reltol = 1e-3,
    abstol = 1e-3
)
    NUM_TYPES = length(model.type_space)

    # We may be using autodiff, so figure out what type the likelikihood value will be
    T = typeof(λ(model, model.type_space[1]))

    # Integrate dp/dt up front.
    # Be sure to specify the iip=true of the ODEProblem for type stability and fewer memory allocations
    p_solution = solve(
        SecondOrderODEProblem{true}(
            dp_dt!,
            ones(Float64, axes(model.type_space)) .- model.ρ,
            zeros(Float64, axes(model.type_space)),
            (0, model.present_time),
            model
        ),
        Tsit5();
        # isoutofdomain for SecondOrderODEProblem passes p::RecursiveArrayTools.ArrayPartition
        # so p.x[1] is p and p.x[2] is integral_p
        isoutofdomain = (p, _, t) -> any(x -> x < 0 || x > 1, p.x[1]),
        # save_everystep = false,
        # save_start = false,
        reltol = reltol,
        abstol = abstol
    )

    """
    Returns a vector of ``p_x(t)`` for type x=`type`.

    Here, `t` is time according to `TreeNode`s (root time is 0, time increases toward leaves)
    and not according to phylogenetics (present time is 0, time increases toward root).
    """
    function p(t, type)
        # OrdinaryDiffEq needs to solve in the increasing time direction, so I need to invert
        t = model.present_time - t

        i = findfirst(==(type), model.type_space)
        
        # Also a RecursiveArrayTools.ArrayPartition
        return p_solution(t; idxs=i)
    end

    """
    Returns a vector of ``\\int_{t_start}^{t_end} p_x(t) dt`` for type x=`type`.

    Here, `t_start` and `t_end` are times according to `TreeNode`s (root time is 0, time increases toward leaves)
    and not according to phylogenetics (present time is 0, time increases toward root).
    """
    function integral_p(t_start, t_end, type)
        # OrdinaryDiffEq needs to solve in the increasing time direction, so I need to invert
        t_start, t_end = model.present_time - t_start, model.present_time - t_end

        i = findfirst(==(type), model.type_space)
        
        # Also a RecursiveArrayTools.ArrayPartition
        l, u = p_solution([t_start, t_end]; idxs=NUM_TYPES+i)

        return u - l
    end

    # Compute logq up each branch (from leaves to root) in the tree
    logq_start = Dict{TreeNode, T}()
    logq_end = Dict{TreeNode, T}()
    
    for event in PostOrderTraversal(tree.children[1])
        parent_type = event.up.type
        λₓ = λ(model, parent_type)
        μₓ = μ(model, parent_type)
        γₓ = γ(model, parent_type)
        
        # Determine initial conditions of branch
        if event.event == :sampled_survival
            logq_end[event] = log(model.ρ)
        elseif event.event == :sampled_death
            logq_end[event] = log(μₓ) + log(model.σ)
        elseif event.event == :birth
            logq_end[event] = (
                log(λₓ)
                + logq_start[event.children[1]]
                + logq_start[event.children[2]]
            )
        elseif event.event == :type_change
            if event.type == event.up.type
                @warn "Self-loop encountered at a type change event in the tree. Density will evaluate to zero."
                return -Inf
            end

            logq_end[event] = (
                log(γ(model, event.up.type, event.type))
                + logq_start[event.children[1]]
            )
        else
            throw(ArgumentError("Unknown event type $(event.event)"))
        end

        # Integral of dq_dt over this branch (up towards the root)
        logq_start[event] = (
            logq_end[event]
            - (λₓ + μₓ + γₓ) * (event.time - event.up.time)
            + 2 * λₓ * integral_p(event.time, event.up.time, parent_type)
        )
    end

    # Compute the non-extinction probability of the whole tree
    pₓ = p(0, tree.type)

    # Condition likelihood on non-extinction
    return logq_start[tree.children[1]] - log(1 - pₓ)
end

function StatsAPI.loglikelihood(
    model::AbstractBranchingProcess,
    trees;
    reltol = 1e-3,
    abstol = 1e-3
)
    return sum(StatsAPI.loglikelihood(model, tree; reltol=reltol, abstol=abstol) for tree in trees)
end

"""
See equation (1) of this paper:

Barido-Sottani, Joëlle, Timothy G Vaughan, and Tanja Stadler. “A Multitype Birth–Death Model for Bayesian Inference of Lineage-Specific Birth and Death Rates.” Edited by Adrian Paterson. Systematic Biology 69, no. 5 (September 1, 2020): 973–86. https://doi.org/10.1093/sysbio/syaa016.
"""
function dp_dt!(dp, p, integral_p, model::FixedTypeChangeRateBranchingProcess, t)
    for (i, type) in enumerate(model.type_space)
        λₓ = λ(model, type)
        μₓ = μ(model, type)
        γₓ = γ(model, type)

        dp[i] = (
            -(λₓ + μₓ + γₓ) * p[i]
            + μₓ * (1 - model.σ)
            + λₓ * p[i]^2
            + γₓ * sum(model.Π[i, j] * p[j] for j in 1:length(model.type_space))
        )
    end
end

function dp_dt!(dp, p, integral_p, model::VaryingTypeChangeRateBranchingProcess, t)
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
    root = TreeNode(init_type)
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
            filter!(child -> child in has_sampled_descendant, node.children)

            if node.event == :birth && length(node.children) == 1
                filter!(child -> child != node, node.up.children)
                push!(node.up.children, node.children[1])
                node.children[1].up = node.up
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
function mutate!(node::TreeNode, model::FixedTypeChangeRateBranchingProcess)
    if node.type ∉ model.type_space
        throw(ArgumentError("The type of the node must be in the type space of the mutator."))
    end

    transition_probs = model.Π[findfirst(==(node.type), model.type_space), :]
    node.type = sample(model.type_space, Weights(transition_probs))
end

function mutate!(node::TreeNode, model::VaryingTypeChangeRateBranchingProcess)
    if node.type ∉ model.type_space
        throw(ArgumentError("The type of the node must be in the type space of the mutator."))
    end

    i = findfirst(==(node.type), model.type_space)
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

    push!(parent.children, child)
    child.up = parent

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
            filter!(child -> child != node, node.up.children)
            push!(node.up.children, node.children[1])
            node.children[1].up = node.up
        end
    end
end