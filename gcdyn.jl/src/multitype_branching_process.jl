# Methods involving `AbstractBranchingProcess` objects.

expit(x) = 1 / (1 + exp(-x))
sigmoid(x, xscale, xshift, yscale, yshift) = yscale * expit(xscale * (x - xshift)) + yshift

# TODO: document
function λ(model::ConstantRateBranchingProcess, state)
    return model.λ
end

function λ(model::SigmoidalBirthRateBranchingProcess, state)
    return sigmoid(state, model.xscale, model.xshift, model.yscale, model.yshift)
end

function μ(model::AbstractBranchingProcess, state)
    return model.μ
end

function γ(model::AbstractBranchingProcess, state)
    return model.γ
end

# To allow us to broadcast the rate parameter functions
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

        leaf.p_end = p.u[end][:]
    end

    for event in PostOrder(tree.children[1])
        t_start = present_time - event.up.t
        t_end = present_time - event.t

        if event.event == :sampled_survival
            # event already has p_end
            event.q_end = ρ
        elseif event.event == :sampled_death
            # event already has p_end
            event.q_end = μ(model, event.up.state) * σ
        elseif event.event == :birth
            event.p_end = event.children[1].p_start
            event.q_end = (
                λ(model, event.up.state) * event.children[1].q_start * event.children[2].q_start
            )
        elseif event.event == :mutation
            mutation_prob = transition_matrix[findfirst(state_space .== event.up.state), findfirst(state_space .== event.state)]

            event.p_end = event.children[1].p_start
            event.q_end = (
                γ(model, event.up.state)
                * mutation_prob
                * event.children[1].q_start
            )
        else
            throw(ArgumentError("Unknown event type $(event.event)"))
        end

        pq = solve(
            ODEProblem(
                dpq_dt!,
                # TODO: I can probably drop the convert now?
                convert(Vector{Float64}, [event.p_end; event.q_end]),
                (t_end, t_start),
                (model, state_space, event.up.state)
            ),
            Tsit5();
            isoutofdomain = (pq, args, t) -> any(x -> x < 0, pq),
            save_everystep = false,
            reltol = reltol,
            abstol = abstol
        )

        event.p_start = pq.u[end][1:end-1]
        event.q_start = pq.u[end][end]
    end

    result = log(tree.children[1].q_start)

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

# TODO: annotate but don't export
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

# TODO: annotate but don't export
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

Can optionally choose not to `reject_stubs`, equivalent to not conditioning on the root node having at least one sampled descendant.

See also [`TreeNode`](@ref).
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
