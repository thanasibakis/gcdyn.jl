# Methods involving `AbstractBranchingProcess` objects.

"``\\frac{1}{1 + exp(-x)}``"
expit(x) = 1 / (1 + exp(-x))

"``\\frac{\\text{yscale}}{1 + exp(-(\\text{xscale} * (x - \text{{xshift}})))} + \\text{yshift}``"
sigmoid(x, xscale, xshift, yscale, yshift) = yscale * expit(xscale * (x - xshift)) + yshift

"""
```julia
λ(model::AbstractBranchingProcess, state)
```

Evaluates the `model`'s birth rate function at the given `state`.
"""
function λ(::AbstractBranchingProcess, state) end

function λ(model::ConstantRateBranchingProcess, state)
    return model.λ
end

function λ(model::SigmoidalBirthRateBranchingProcess, state)
    return sigmoid(state, model.xscale, model.xshift, model.yscale, model.yshift)
end

"""
```julia
μ(model::AbstractBranchingProcess, state)
```

Evaluates the `model`'s death rate function at the given `state`.
"""
function μ(model::AbstractBranchingProcess, state)
    return model.μ
end

"""
```julia
γ(model::AbstractBranchingProcess, state)
```

Evaluates the `model`'s type change rate function at the given `state`.
"""
function γ(model::AbstractBranchingProcess, state)
    return model.γ
end

function γ(model::VaryingTypeChangeRateBranchingProcess, state)
    i = findfirst_equals(model.state_space, state)

    return model.δ * -model.Q[i, i]
end

"""
```julia
γ(model::AbstractBranchingProcess, from_state, to_state)
```

Evaluates the `model`'s rate of type change from `from_state` to `to_state`.
"""
function γ(model::AbstractBranchingProcess, from_state, to_state)
    if from_state == to_state
        return 0
    end

    i = findfirst_equals(model.state_space, from_state)
    j = findfirst_equals(model.state_space, to_state)

    return model.γ * model.transition_matrix[i, j]
end

function γ(model::VaryingTypeChangeRateBranchingProcess, from_state, to_state)
    if from_state == to_state
        return 0
    end

    i = findfirst_equals(model.state_space, from_state)
    j = findfirst_equals(model.state_space, to_state)

    return model.δ * model.Q[i, j]
end

# To allow us to broadcast the rate parameter functions over states
Base.broadcastable(model::AbstractBranchingProcess) = Ref(model)

"""
```julia
loglikelihood(model::AbstractBranchingProcess, tree::TreeNode; kw...)
loglikelihood(model::AbstractBranchingProcess, trees::AbstractVector{TreeNode}; kw...)
```

A slight deviation from StatsAPI, as observations are not stored in the model object, so they must be passed as an argument.

Keyword arguments `reltol` and `abstol` are passed to the ODE solver.
"""
function StatsAPI.loglikelihood(
    model::AbstractBranchingProcess,
    trees::AbstractVector{TreeNode};
    reltol = 1e-6,
    abstol = 1e-6
)
    return sum(loglikelihood(model, tree; reltol=reltol, abstol=abstol) for tree in trees)
end

function StatsAPI.loglikelihood(
    model::AbstractBranchingProcess,
    tree::TreeNode;
    reltol = 1e-6,
    abstol = 1e-6
)

    ρ, σ = model.ρ, model.σ
    present_time = model.present_time

    # We may be using autodiff, so figure out what type the likelikihood value will be
    T = typeof(λ(model, model.state_space[1]))

    p_start = Dict{TreeNode, Vector{T}}()
    p_end = Dict{TreeNode, Vector{T}}()
    q_start = Dict{TreeNode, T}()
    q_end = Dict{TreeNode, T}()

    for leaf in LeafTraversal(tree)
        # Be sure to specify the iip=true of the ODEProblem for type stability
        # and fewer memory allocations
        p = solve(
            ODEProblem{true}(
                dp_dt!,
                ones(Float64, axes(model.state_space)) .- ρ,
                (0, present_time - leaf.t),
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
        t_start = present_time - event.up.t
        t_end = present_time - event.t

        if event.event == :sampled_survival
            # event already has p_end
            q_end[event] = ρ
        elseif event.event == :sampled_death
            # event already has p_end
            q_end[event] = μ(model, event.up.state) * σ
        elseif event.event == :birth
            p_end[event] = p_start[event.children[1]]
            q_end[event] = (
                λ(model, event.up.state) * q_start[event.children[1]] * q_start[event.children[2]]
            )
        elseif event.event == :mutation
            p_end[event] = p_start[event.children[1]]
            q_end[event] = (
                γ(model, event.up.state, event.state)
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
                (model, event.up.state)
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
            ones(Float64, axes(model.state_space)) .- ρ,
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

    p_i = p.u[end][findfirst_equals(model.state_space, tree.state)]
    result -= log(1 - p_i)

    return result
end

"""
See equation (1) of this paper:

Barido-Sottani, Joëlle, Timothy G Vaughan, and Tanja Stadler. “A Multitype Birth–Death Model for Bayesian Inference of Lineage-Specific Birth and Death Rates.” Edited by Adrian Paterson. Systematic Biology 69, no. 5 (September 1, 2020): 973–86. https://doi.org/10.1093/sysbio/syaa016.
"""
function dp_dt!(dp, p, model::AbstractBranchingProcess, t)
    for (i, state) in enumerate(model.state_space)
        λₓ = λ(model, state)
        μₓ = μ(model, state)
        γₓ = γ(model, state)

        dp[i] = (
            -(λₓ + μₓ + γₓ) * p[i]
            + μₓ * (1 - model.σ)
            + λₓ * p[i]^2
            + γₓ * sum(model.transition_matrix[i, j] * p[j] for j in 1:length(model.state_space))
        )
    end
end

function dp_dt!(dp, p, model::VaryingTypeChangeRateBranchingProcess, t)
    for (i, state) in enumerate(model.state_space)
        λₓ = λ(model, state)
        μₓ = μ(model, state)

        dp[i] = (
            -(λₓ + μₓ) * p[i]
            + μₓ * (1 - model.σ)
            + λₓ * p[i]^2
            + model.δ * sum(model.Q[i, j] * p[j] for j in 1:length(model.state_space))
        )
    end
end

"""
See equations (1) and (2) of this paper:

Barido-Sottani, Joëlle, Timothy G Vaughan, and Tanja Stadler. “A Multitype Birth–Death Model for Bayesian Inference of Lineage-Specific Birth and Death Rates.” Edited by Adrian Paterson. Systematic Biology 69, no. 5 (September 1, 2020): 973–86. https://doi.org/10.1093/sysbio/syaa016.
"""
function dpq_dt!(dpq, pq, args, t)
    p, q_i = view(pq, 1:lastindex(pq)-1), pq[end]
    model, parent_state = args

    λₓ = λ(model, parent_state)
    μₓ = μ(model, parent_state)
    γₓ = γ(model, parent_state)

    p_i = p[findfirst_equals(model.state_space, parent_state)]
    dq_i = -(λₓ + μₓ + γₓ) * q_i + 2 * λₓ * q_i * p_i

    # Need to pass a view instead of a slice, to pass by reference instead of value
    dp = view(dpq, 1:lastindex(dpq)-1)
    dp_dt!(dp, p, model, t)
    dpq[end] = dq_i
end

"""
```julia
findfirst_equals(x, c)
```

Like `findfirst(x .== c)`, but avoids making memory allocations.

See also [`findfirst`](@ref).
"""
function findfirst_equals(x, c)
    for i in 1:length(x)
        if x[i] == c
            return i
        end
    end

    return -1
end

"""
```julia
rand_tree(model, init_state; reject_stubs=true)
rand_tree(model, n, init_state; reject_stubs=true)
```

Generate `n` random trees from the given model (optional parameter, default is `1`), each starting at the given initial state.

Note that trees will have root time `0`, with time increasing toward the tips.
This aligns us with the branching process models, but contradicts the notion of time typically used in phylogenetics.

Can optionally choose not to `reject_stubs`, equivalent to not conditioning on the root node having at least one sampled descendant.

See also [`TreeNode`](@ref), [`sample_child!`](@ref), [`mutate!`](@ref).
"""
function rand_tree(
    model::AbstractBranchingProcess,
    n::Int,
    init_state::Real;
    reject_stubs::Bool=true
)
    trees = Vector{TreeNode}(undef, n)

    for i in 1:n
        trees[i] = rand_tree(model, init_state; reject_stubs=reject_stubs)
    end

    return trees
end

function rand_tree(
    model::AbstractBranchingProcess,
    init_state::Real;
    reject_stubs::Bool=true
)
    root = TreeNode(init_state)
    needs_children = [root]

    # Evolve a fully-observed tree
    while length(needs_children) > 0
        node = pop!(needs_children)
        child = sample_child!(node, model)

        if child.event == :birth
            push!(needs_children, child)
            push!(needs_children, child)
        elseif child.event == :mutation
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
        return rand_tree(model, init_state; reject_stubs = true)
    end

    return root
end

"""
```julia
mutate!(node, model)
```

Change the state of the given node according to the transition probabilities in the given model.
"""
function mutate!(node::TreeNode, model::AbstractBranchingProcess)
    if node.state ∉ model.state_space
        throw(ArgumentError("The state of the node must be in the state space of the mutator."))
    end

    transition_probs = model.transition_matrix[findfirst_equals(model.state_space, node.state), :]
    node.state = sample(model.state_space, Weights(transition_probs))
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

    λₓ, μₓ, γₓ = λ(model, parent.state), μ(model, parent.state), γ(model, parent.state)

    waiting_time = rand(Exponential(1 / (λₓ + μₓ + γₓ)))

    if parent.t + waiting_time > model.present_time
        event = sample([:sampled_survival, :unsampled_survival], Weights([model.ρ, 1 - model.ρ]))
        child = TreeNode(event, model.present_time, parent.state)
    else
        event = sample([:birth, :unsampled_death, :mutation], Weights([λₓ, μₓ, γₓ]))

        if event == :unsampled_death && rand() < model.σ
            event = :sampled_death
        end

        child = TreeNode(event, parent.t + waiting_time, parent.state)

        if event == :mutation
            mutate!(child, model)
        end
    end

    push!(parent.children, child)
    child.up = parent

    return child
end

"""
```julia
discretize_states(trees)
```

Replaces a continuous-valued state attribute of all nodes in `trees` with a discretized version.

`state_space_size` bins are created from evenly spaced quantiles of the state distribution over all trees, and states
are discretized to the medians of their bins.
Optionally but by default, if the discretization process results in a mutation event with the same state as its parent,
the mutation event is pruned from the tree.
"""
function discretize_states!(trees, state_space_size; prune_self_loops = true)
    all_nodes = [node for tree in trees for node in PostOrderTraversal(tree)]

    # Create bins from evenly spaced quantiles, then discretize states to the medians of their bins
    states = DataFrame(state=[node.state for node in all_nodes])
    cutoffs = quantile(states.state, 0:(1/state_space_size):1)
    states.bin = cut(states.state, cutoffs; extend=true)
    bin_map = transform(groupby(states, :bin), :state => median => :binned_state)
    binned_states = bin_map.binned_state
    
    for (node, binned_state) in zip(all_nodes, binned_states)
        node.state = binned_state
    end

    if prune_self_loops
        for tree in trees
            # If a mutation resulted in an state of the same bin as the parent,
            # that isn't a valid type change in the CTMC, so we prune it

            for node in PostOrderTraversal(tree)
                if node.event == :mutation && node.state == node.up.state
                    filter!(child -> child != node, node.up.children)
                    push!(node.up.children, node.children[1])
                    node.children[1].up = node.up
                end
            end
        end
    end

    bin_map = select(bin_map, Not(:state)) |> unique
    parse_interval(row) = parse.(Float64, split(convert(String, row.bin)[2:end-1], ", "))

    return Dict(parse_interval(row) => row.binned_state for row in eachrow(bin_map))
end
