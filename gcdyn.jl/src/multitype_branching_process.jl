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

# To allow us to broadcast the rate parameter functions over states
Base.length(::AbstractBranchingProcess) = 1
Base.iterate(model::AbstractBranchingProcess) = (model, nothing)
Base.iterate(::AbstractBranchingProcess, state) = nothing

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
    state_space = model.state_space
    transition_matrix = model.transition_matrix
    present_time = model.present_time

    for leaf in Leaves(tree)
        p = solve(
            ODEProblem(
                dp_dt!,
                ones(Float64, axes(state_space)) .- ρ,
                (0, present_time - leaf.t),
                (model, state_space)
            ),
            Tsit5();
            isoutofdomain = (p, args, t) -> any(x -> x < 0 || x > 1, p),
            save_everystep = false,
            reltol = reltol,
            abstol = abstol
        )

        leaf.info[:p_end] = p.u[end][:]
    end

    for event in PostOrderTraversal(tree.children[1])
        t_start = present_time - event.up.t
        t_end = present_time - event.t

        if event.event == :sampled_survival
            # event already has p_end
            event.info[:q_end] = ρ
        elseif event.event == :sampled_death
            # event already has p_end
            event.info[:q_end] = μ(model, event.up.state) * σ
        elseif event.event == :birth
            event.info[:p_end] = event.children[1].info[:p_start]
            event.info[:q_end] = (
                λ(model, event.up.state) * event.children[1].info[:q_start] * event.children[2].info[:q_start]
            )
        elseif event.event == :mutation
            mutation_prob = transition_matrix[findfirst(state_space .== event.up.state), findfirst(state_space .== event.state)]

            event.info[:p_end] = event.children[1].info[:p_start]
            event.info[:q_end] = (
                γ(model, event.up.state)
                * mutation_prob
                * event.children[1].info[:q_start]
            )
        else
            throw(ArgumentError("Unknown event type $(event.event)"))
        end

        pq = solve(
            ODEProblem(
                dpq_dt!,
                # TODO: I can probably drop the convert now?
                convert(Vector{Float64}, [event.info[:p_end]; event.info[:q_end]]),
                (t_end, t_start),
                (model, state_space, event.up.state)
            ),
            Tsit5();
            isoutofdomain = (pq, args, t) -> any(x -> x < 0, pq),
            save_everystep = false,
            reltol = reltol,
            abstol = abstol
        )

        event.info[:p_start] = pq.u[end][1:end-1]
        event.info[:q_start] = pq.u[end][end]
    end

    result = log(tree.children[1].info[:q_start])

    # Non-extinction probability

    p = solve(
        ODEProblem(
            dp_dt!,
            ones(Float64, axes(state_space)) .- ρ,
            (0, present_time),
            (model, state_space)
        ),
        Tsit5();
        isoutofdomain = (p, args, t) -> any(x -> x < 0 || x > 1, p),
        save_everystep = false,
        reltol = reltol,
        abstol = abstol
    )

    p_i::Float64 = p.u[end][findfirst(state_space .== tree.state)]
    result -= log(1 - p_i)

    return result
end

"""
See equation (1) of this paper:

Barido-Sottani, Joëlle, Timothy G Vaughan, and Tanja Stadler. “A Multitype Birth–Death Model for Bayesian Inference of Lineage-Specific Birth and Death Rates.” Edited by Adrian Paterson. Systematic Biology 69, no. 5 (September 1, 2020): 973–86. https://doi.org/10.1093/sysbio/syaa016.
"""
function dp_dt!(dp, p, args, t)
    model, state_space = args

    λₓ::Vector{Float64} = λ.(model, state_space)
    μₓ::Vector{Float64} = μ.(model, state_space)
    γₓ::Vector{Float64} = γ.(model, state_space)
    σ = model.σ
    transition_matrix = model.transition_matrix

    dp[:] = (
        -(γₓ + λₓ + μₓ) .* p
        + μₓ * (1 - σ)
        + λₓ .* p.^2
        + γₓ .* (transition_matrix * p)
    )
end

"""
See equations (1) and (2) of this paper:

Barido-Sottani, Joëlle, Timothy G Vaughan, and Tanja Stadler. “A Multitype Birth–Death Model for Bayesian Inference of Lineage-Specific Birth and Death Rates.” Edited by Adrian Paterson. Systematic Biology 69, no. 5 (September 1, 2020): 973–86. https://doi.org/10.1093/sysbio/syaa016.
"""
function dpq_dt!(dpq, pq, args, t)
    p, q_i = pq[1:end-1], pq[end]
    model, state_space, parent_state = args

    λₓ::Float64 = λ(model, parent_state)
    μₓ::Float64 = μ(model, parent_state)
    γₓ::Float64 = γ(model, parent_state)

    dq_i = -(γₓ + λₓ + μₓ) * q_i + 2 * λₓ * q_i * p[findfirst(state_space .== parent_state)]

    # Need to pass a view instead of a slice, to pass by reference instead of value
    dp_dt!(view(dpq, 1:lastindex(dpq)-1), p, (model, state_space), t)
    dpq[end] = dq_i
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

    transition_probs = model.transition_matrix[findfirst(model.state_space .== node.state), :]
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