# Methods involving `TreeNode` objects.

# Implemented to get access to AbstractTrees API
AbstractTrees.parent(node::TreeNode) = node.up
AbstractTrees.children(node::TreeNode) = node.children
AbstractTrees.nodevalue(node::TreeNode) = node.state
AbstractTrees.childtype(::TreeNode) = TreeNode
AbstractTrees.ChildIndexing(::TreeNode) = AbstractTrees.IndexedChildren()
AbstractTrees.NodeType(::TreeNode) = AbstractTrees.HasNodeType()
AbstractTrees.nodetype(::TreeNode) = TreeNode

Base.show(io::IO, tree::TreeNode) = AbstractTrees.print_tree(io, tree)

"""
```julia
mutate!(node, model)
```

Change the state of the given node according to the transition probabilities in the given model.
"""
function mutate!(node::TreeNode, model::MultitypeBranchingProcess)
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
"""
function sample_child!(parent::TreeNode, model::MultitypeBranchingProcess)
    if length(parent.children) == 2
        throw(ArgumentError("Can only have 2 children max"))
    end

    λₓ, μₓ, γₓ = model.λ(parent.state), model.μ(parent.state), model.γ(parent.state)

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