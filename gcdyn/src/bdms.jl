MAX_CHILDREN = 2
EVENTS = [:root, :birth, :sampled_death, :unsampled_death, :mutation, :sampled_survival, :unsampled_survival]

mutable struct TreeNode
    event::Symbol
    t::Real
    phenotype::Any
    children::Vector{TreeNode}
    up::Union{TreeNode, Nothing}
    has_sampled_descendant::Bool

    function TreeNode(event, t, phenotype, children, up, has_sampled_descendant)
        if event ∉ EVENTS
            throw(ArgumentError("Event must be one of $(EVENTS)"))
        elseif t < 0
            throw(ArgumentError("Time must be positive"))
        elseif length(children) > MAX_CHILDREN
            throw(ArgumentError("Can only have $MAX_CHILDREN children max"))
        end

        return new(event, t, phenotype, children, up, has_sampled_descendant)
    end
end

function TreeNode(event, t, phenotype)
    return TreeNode(event, t, phenotype, [], nothing, false)
end

function TreeNode(phenotype)
    return TreeNode(:root, 0, phenotype, [], nothing, false)
end

abstract type AbstractMutator end

struct DiscreteMutator <: AbstractMutator
    state_space::Vector
    transition_matrix::Matrix{Real}

    function DiscreteMutator(state_space, transition_matrix)
        if length(state_space) < 2
            throw(ArgumentError("The state space must contain at least two states."))
        elseif length(state_space) != size(transition_matrix, 1)
            throw(DimensionMismatch("The number of states in the state space must match the number of rows in the transition matrix."))
        elseif length(state_space) != size(transition_matrix, 2)
            throw(DimensionMismatch("The number of states in the state space must match the number of columns in the transition matrix."))
        elseif any(transition_matrix .< 0) || any(transition_matrix .> 1)
            throw(ArgumentError("The transition matrix must contain only values between 0 and 1."))
        elseif any(sum(transition_matrix, dims=2) .!= 1)
            throw(ArgumentError("The transition matrix must contain only rows that sum to 1."))
        elseif any(!=(0), transition_matrix[i,i] for i in minimum(axes(transition_matrix)))
            throw(ArgumentError("The transition matrix must contain only zeros on the diagonal."))
        end

        new(state_space, transition_matrix)
    end
end

function DiscreteMutator(state_space::AbstractVector)
    transition_matrix = (ones(length(state_space), length(state_space)) - I) / (length(state_space) - 1)

    return DiscreteMutator(state_space, transition_matrix)
end

# Implemented to get access to AbstractTrees API
AbstractTrees.parent(node::TreeNode) = node.up
AbstractTrees.children(node::TreeNode) = node.children
AbstractTrees.nodevalue(node::TreeNode) = node.phenotype
AbstractTrees.childtype(::TreeNode) = TreeNode
AbstractTrees.ChildIndexing(::TreeNode) = AbstractTrees.IndexedChildren()
AbstractTrees.NodeType(::TreeNode) = AbstractTrees.HasNodeType()
AbstractTrees.nodetype(::TreeNode) = TreeNode

Base.show(io::IO, tree::TreeNode) = AbstractTrees.print_tree(io, tree)

function evolve!(
    root::TreeNode,
    time::Real,
    λ::Function,
    μ::Function,
    γ::Function,
    mutator::AbstractMutator,
    ρ::Real,
    σ::Real
)
    needs_children = [root]

    while length(needs_children) > 0
        node = pop!(needs_children)
        child = sample_child!(node, λ, μ, γ, mutator, ρ, σ, time)

        if child.event == :birth
            for _ in 1:MAX_CHILDREN
                push!(needs_children, child)
            end
        elseif child.event == :mutation
            push!(needs_children, child)
        end
    end

    # Now we flag every node that has a sampled descendant
    for leaf in AbstractTrees.Leaves(root)
        if leaf.event ∉ (:sampled_survival, :sampled_death)
            continue
        end

        node = leaf

        while !isnothing(node)
            node.has_sampled_descendant = true
            node = node.up
        end
    end
end

function prune!(tree::TreeNode)
    for node in AbstractTrees.PostOrderDFS(tree)
        if node.event ∈ (:birth, :root)
            filter!(child -> child.has_sampled_descendant, node.children)

            if node.event == :birth && length(node.children) == 1
                filter!(child -> child != node, node.up.children)
                push!(node.up.children, node.children[1])
                node.children[1].up = node.up
            end
        end
    end
end

function sample_child!(parent::TreeNode, λ::Function, μ::Function, γ::Function, mutator::AbstractMutator, ρ::Real, σ::Real, max_time::Real)
    if length(parent.children) == MAX_CHILDREN
        throw(ArgumentError("Can only have $MAX_CHILDREN children max"))
    end

    λₓ, μₓ, γₓ = λ(parent.phenotype), μ(parent.phenotype), γ(parent.phenotype)

    waiting_time = rand(Exponential(1 / (λₓ + μₓ + γₓ)))

    if parent.t + waiting_time > max_time
        event = sample([:sampled_survival, :unsampled_survival], Weights([ρ, 1 - ρ]))
        child = TreeNode(event, max_time, parent.phenotype)
    else
        event = sample([:birth, :unsampled_death, :mutation], Weights([λₓ, μₓ, γₓ]))

        if event == :unsampled_death && rand() < σ
            event = :sampled_death
        end

        child = TreeNode(event, parent.t + waiting_time, parent.phenotype)

        if event == :mutation
            mutate!(mutator, child)
        end
    end

    push!(parent.children, child)
    child.up = parent

    return child
end

function mutate!(mutator::DiscreteMutator, node::TreeNode)
    if node.phenotype ∉ mutator.state_space
        throw(ArgumentError("The phenotype of the node must be in the state space of the mutator."))
    end

    transition_probs = mutator.transition_matrix[findfirst(mutator.state_space .== node.phenotype), :]
    node.phenotype = sample(mutator.state_space, Weights(transition_probs))
end

function Distributions.logpdf(mutator::DiscreteMutator, node::TreeNode)
    if node.phenotype ∉ mutator.state_space || node.up.phenotype ∉ mutator.state_space
        throw(ArgumentError("The phenotype of the node must be in the state space of the mutator."))
    end

    return log(mutator.transition_matrix[findfirst(mutator.state_space .== node.up.phenotype), findfirst(mutator.state_space .== node.phenotype)])
end
