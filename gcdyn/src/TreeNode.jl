# Defines the TreeNode type and implements the AbstractTrees API

EVENTS = [:root, :birth, :sampled_death, :unsampled_death, :mutation, :sampled_survival, :unsampled_survival]

# TODO: make this immutable and adjust mutators somehow
mutable struct TreeNode
    event::Symbol
    t::Real
    phenotype::Real
    children::Vector{TreeNode}
    up::Union{TreeNode, Nothing}

    function TreeNode(event, t, phenotype, children, up)
        if event ∉ EVENTS
            throw(ArgumentError("Event must be one of $(EVENTS)"))
        elseif t < 0
            throw(ArgumentError("Time must be positive"))
        elseif length(children) > 2
            throw(ArgumentError("Can only have 2 children max"))
        end

        return new(event, t, phenotype, children, up)
    end
end

function TreeNode(event, t, phenotype)
    return TreeNode(event, t, phenotype, [], nothing)
end

function TreeNode(phenotype)
    return TreeNode(:root, 0, phenotype, [], nothing)
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
