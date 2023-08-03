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
    ρ::Real,
    σ::Real
)
    needs_children = [root]

    while length(needs_children) > 0
        node = pop!(needs_children)
        child = sample_child!(node, λ, μ, γ, ρ, σ, time)

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

function sample_child!(parent::TreeNode, λ::Function, μ::Function, γ::Function, ρ::Real, σ::Real, max_time::Real)
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
    end

    push!(parent.children, child)
    child.up = parent

    return child
end