# Defines the MultitypeBranchingProcess type and implements rand_tree, Distributions.logpdf, StatsBase.loglikelihood

struct MultitypeBranchingProcess
    λ::Function
    μ::Function
    γ::Function
    state_space::AbstractVector
    transition_matrix::AbstractMatrix
    ρ::Real
    σ::Real
    present_time::Real

    function MultitypeBranchingProcess(λ::Union{Real, Function}, μ::Union{Real, Function}, γ::Union{Real, Function}, state_space, transition_matrix, ρ, σ, present_time)
        if ρ < 0 || ρ > 1
            throw(ArgumentError("ρ must be between 0 and 1"))
        elseif σ < 0 || σ > 1
            throw(ArgumentError("σ must be between 0 and 1"))
        elseif present_time < 0
            throw(ArgumentError("Time must be positive"))
        elseif length(state_space) != size(transition_matrix, 1)
            throw(DimensionMismatch("The number of states in the state space must match the number of rows in the transition matrix."))
        elseif length(state_space) != size(transition_matrix, 2)
            throw(DimensionMismatch("The number of states in the state space must match the number of columns in the transition matrix."))
        elseif any(transition_matrix .< 0) || any(transition_matrix .> 1)
            throw(ArgumentError("The transition matrix must contain only values between 0 and 1."))
        elseif any(sum(transition_matrix, dims=2) .!= 1)
            throw(ArgumentError("The transition matrix must contain only rows that sum to 1."))
        elseif any(!=(0), transition_matrix[i,i] for i in minimum(axes(transition_matrix))) && length(state_space) > 1
            throw(ArgumentError("The transition matrix must contain only zeros on the diagonal."))
        end

        # Convert scalars to constant functions
        λ⁺::Function = isa(λ, Function) ? λ : _ -> λ
        μ⁺::Function = isa(μ, Function) ? μ : _ -> μ
        γ⁺::Function = isa(γ, Function) ? γ : _ -> γ

        # Ensure functions return floating point numbers
        λ⁺⁺ = x -> convert(AbstractFloat, λ⁺(x))
        μ⁺⁺ = x -> convert(AbstractFloat, μ⁺(x))
        γ⁺⁺ = x -> convert(AbstractFloat, γ⁺(x))

        return new(λ⁺⁺, μ⁺⁺, γ⁺⁺, state_space, transition_matrix, ρ, σ, present_time)
    end
end

# Uniform transition matrix
function MultitypeBranchingProcess(λ::Union{Real, Function}, μ::Union{Real, Function}, γ::Union{Real, Function}, state_space::AbstractVector, ρ::Real, σ::Real, present_time::Real)
    transition_matrix = (ones(length(state_space), length(state_space)) - I) / (length(state_space) - 1)

    return MultitypeBranchingProcess(λ, μ, γ, state_space, transition_matrix, ρ, σ, present_time)
end

function StatsBase.loglikelihood(
    model::MultitypeBranchingProcess,
    trees::AbstractVector{TreeNode}
)
    return sum(logpdf(model, tree) for tree in trees)
end

function Distributions.logpdf(model::MultitypeBranchingProcess, tree::TreeNode; reltol = 1e-6, abstol = 1e-6)
    λ = model.λ
    μ = model.μ
    γ = model.γ
    state_space = model.state_space
    transition_matrix = model.transition_matrix
    ρ = model.ρ
    σ = model.σ
    present_time = model.present_time

    function dp_dt!(dp, p, _, t)
        dp[:] = (
            -(
                γ.(state_space)
                + λ.(state_space)
                + μ.(state_space)
            )
            .* p
            + μ.(state_space) * (1 - σ)
            + λ.(state_space) .* p.^2
            + γ.(state_space) .* (transition_matrix * p)
        )
    end

    function dpq_dt!(dpq, pq, args, t)
        p, q_i = pq[1:end-1], pq[end]
        parent_phenotype = args

        dq_i = -(
            γ(parent_phenotype)
            + λ(parent_phenotype)
            + μ(parent_phenotype)
        ) * q_i + 2 * λ(parent_phenotype) * q_i * p[findfirst(state_space .== parent_phenotype)]

        # Need to pass a view instead of a slice, to pass by reference instead of value
        dp_dt!(view(dpq, 1:lastindex(dpq)-1), p, nothing, t)
        dpq[end] = dq_i
    end

    p_start, p_end = Dict{TreeNode, Vector{AbstractFloat}}(), Dict{TreeNode, Vector{AbstractFloat}}()
    q_start, q_end = Dict{TreeNode, AbstractFloat}(), Dict{TreeNode, AbstractFloat}()

    for leaf in AbstractTrees.Leaves(tree)
        p = solve(
            ODEProblem(
                dp_dt!,
                ones(AbstractFloat, axes(state_space)) .- ρ,
                (0, present_time - leaf.t)
            ),
            Tsit5();
            save_everystep = false,
            reltol = reltol,
            abstol = abstol
        )

        p_end[leaf] = p.u[end][:]
    end

    for event in AbstractTrees.PostOrderDFS(tree.children[1])
        t_start = present_time - event.up.t
        t_end = present_time - event.t

        if event.event == :sampled_survival
            # event already has p_end
            q_end[event] = ρ
        elseif event.event == :sampled_death
            # event already has p_end
            q_end[event] = μ(event.up.phenotype) * σ
        elseif event.event == :birth
            p_end[event] = p_start[event.children[1]]
            q_end[event] = (
                λ(event.up.phenotype) * q_start[event.children[1]] * q_start[event.children[2]]
            )
        elseif event.event == :mutation
            mutation_prob = transition_matrix[findfirst(state_space .== event.up.phenotype), findfirst(state_space .== event.phenotype)]

            p_end[event] = p_start[event.children[1]]
            q_end[event] = (
                γ(event.up.phenotype)
                * mutation_prob
                * q_start[event.children[1]]
            )
        else
            throw(ArgumentError("Unknown event type $(event.event)"))
        end

        pq = solve(
            ODEProblem(
                dpq_dt!,
                # TODO: this must convert to Float64, not even AbstractFloat. Why?
                convert(Vector{Float64}, [p_end[event]; q_end[event]]),
                (t_end, t_start),
                event.up.phenotype
            ),
            Tsit5();
            save_everystep = false,
            reltol = reltol,
            abstol = abstol
        )

        p_start[event] = pq.u[end][1:end-1]
        q_start[event] = pq.u[end][end]
    end

    result = log(q_start[tree.children[1]])

    # Non-extinction probability

    p = solve(
        ODEProblem(
            dp_dt!,
            ones(AbstractFloat, axes(state_space)) .- ρ,
            (0, present_time)
        ),
        Tsit5();
        save_everystep = false,
        reltol = reltol,
        abstol = abstol
    )

    p_i = p.u[end][findfirst(state_space .== tree.phenotype)]
    result -= log(1 - p_i)

    return result
end

function rand_tree(
    model::MultitypeBranchingProcess,
    n::Int,
    init_phenotype::Real;
    reject_stubs::Bool=true
)
    trees = Vector{TreeNode}(undef, n)

    for i in 1:n
        trees[i] = rand_tree(model, init_phenotype; reject_stubs=reject_stubs)
    end

    return trees
end

function rand_tree(
    model::MultitypeBranchingProcess,
    init_phenotype::Real;
    reject_stubs::Bool=true
)
    root = TreeNode(init_phenotype)
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

    for leaf in AbstractTrees.Leaves(root)
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
    for node in AbstractTrees.PostOrderDFS(root)
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
        return rand_tree(model, init_phenotype; reject_stubs = true)
    end

    return root
end

function mutate!(node::TreeNode, model::MultitypeBranchingProcess)
    if node.phenotype ∉ model.state_space
        throw(ArgumentError("The phenotype of the node must be in the state space of the mutator."))
    end

    transition_probs = model.transition_matrix[findfirst(model.state_space .== node.phenotype), :]
    node.phenotype = sample(model.state_space, Weights(transition_probs))
end

function sample_child!(parent::TreeNode, model::MultitypeBranchingProcess)
    if length(parent.children) == 2
        throw(ArgumentError("Can only have 2 children max"))
    end

    λₓ, μₓ, γₓ = model.λ(parent.phenotype), model.μ(parent.phenotype), model.γ(parent.phenotype)

    waiting_time = rand(Exponential(1 / (λₓ + μₓ + γₓ)))

    if parent.t + waiting_time > model.present_time
        event = sample([:sampled_survival, :unsampled_survival], Weights([model.ρ, 1 - model.ρ]))
        child = TreeNode(event, model.present_time, parent.phenotype)
    else
        event = sample([:birth, :unsampled_death, :mutation], Weights([λₓ, μₓ, γₓ]))

        if event == :unsampled_death && rand() < model.σ
            event = :sampled_death
        end

        child = TreeNode(event, parent.t + waiting_time, parent.phenotype)

        if event == :mutation
            mutate!(child, model)
        end
    end

    push!(parent.children, child)
    child.up = parent

    return child
end