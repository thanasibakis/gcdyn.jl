# Methods involving `TreeNode` objects.

"""
```julia
attach!(parent::TreeNode, child::TreeNode)
```

Adds `child` to `parent.children` and sets `child.up` to `parent`.
"""
function attach!(parent::TreeNode, child::TreeNode)
    push!(parent.children, child)
    child.up = parent

    return nothing
end

"""
```julia
detach!(parent::TreeNode, child::TreeNode)
```

Removes `child` from `parent.children` and sets `child.up` to `nothing`.
"""
function detach!(parent::TreeNode, child::TreeNode)
    filter!(!=(child), parent.children)
    child.up = nothing

    return nothing
end

"""
```julia
delete!(node::TreeNode)
```

Removes this node from its place in the tree, attaching its children to its parent.
"""
function Base.delete!(node::TreeNode)
    parent = node.up
    children = collect(node.children) # not a pointer

    detach!(parent, node)

    for child in children
        detach!(node, child)
        attach!(parent, child)
    end
end

"""
```julia
map_types(tree, mapping; prune_self_loops = true)
```

Returns a new tree like `tree`, but replaces the type attribute of each node with the result of the callable `mapping` applied to the type value.

Optionally but by default, if the map results in a type change event with the same type as its parent,
the type change event is pruned from the tree.
"""
function map_types(mapping, tree; prune_self_loops = true)
    # The mapping may change the type T of a TreeNode, so we can't just traverse and update node.type

    function helper!(mapping, new_node, old_node)
        for old_child in old_node.children
            new_child = TreeNode(old_child.event, old_child.time, mapping(old_child.type))
            attach!(new_node, helper!(mapping, new_child, old_child))
        end

        return new_node
    end

    new_tree = TreeNode(tree.event, tree.time, mapping(tree.type))
    helper!(mapping, new_tree, tree)

    # If a type change resulted in an type of the same bin as the parent,
    # that isn't a valid type change in the CTMC, so we prune it
    for node in PreOrderTraversal(new_tree)
        if prune_self_loops && node.event == :type_change && node.type == node.up.type
            delete!(node)
        end
    end

    return new_tree
end

"""
```julia
map_types!(tree, mapping; prune_self_loops = true)
```

Replaces the type attribute of all nodes in `tree` with the result of the callable `mapping` applied to the type value.

Note that `TreeNode{T}` objects have a fixed `typeof(node.type) == T`, so the `mapping` must be compliant with this.

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


"""
```julia
TreeTraversal
```
Abstract supertype for iterators over `TreeNode`s.

Pre-order and post-order traversals are defined.
These are present mainly because the iterators in `AbstractTrees` were difficult to make type-stable.

See also [`PostOrderTraversal`](@ref), [`PreOrderTraversal`](@ref).
"""
abstract type TreeTraversal end

"""
```julia
PostOrderTraversal(tree)
```

Iterator for performing a post-order traversal over `tree`.

See also [`TreeTraversal`](@ref).
"""
struct PostOrderTraversal <: TreeTraversal
    root::TreeNode
end

"""
```julia
PreOrderTraversal(tree)
```

Iterator for performing a pre-order traversal over `tree`.
"""
struct PreOrderTraversal <: TreeTraversal
    root::TreeNode
end

"""
```julia
LeafTraversal(tree)
```

Iterator over the leaves of `tree`.
"""
struct LeafTraversal <: TreeTraversal
    root::TreeNode
end

function Base.iterate(it::PostOrderTraversal)
    if isnothing(it.root)
        return nothing
    end

    function collect_nodes!(nodes, tree)
        for child in tree.children
            collect_nodes!(nodes, child)
        end
        push!(nodes, tree)
    end

    to_visit = Vector{TreeNode}()
    collect_nodes!(to_visit, it.root)

    return popfirst!(to_visit), to_visit
end

function Base.iterate(it::PreOrderTraversal)
    if isnothing(it.root)
        return nothing
    end

    function collect_nodes!(nodes, tree)
        push!(nodes, tree)

        for child in tree.children
            collect_nodes!(nodes, child)
        end
    end

    to_visit = Vector{TreeNode}()
    collect_nodes!(to_visit, it.root)

    return popfirst!(to_visit), to_visit
end

function Base.iterate(it::LeafTraversal)
    if isnothing(it.root)
        return nothing
    end

    function collect_nodes!(nodes, tree)
        if length(tree.children) == 0
            push!(nodes, tree)
        else
            for child in tree.children
                collect_nodes!(nodes, child)
            end
        end
    end

    to_visit = Vector{TreeNode}()
    collect_nodes!(to_visit, it.root)

    return popfirst!(to_visit), to_visit
end

function Base.iterate(::TreeTraversal, to_visit)
    if isempty(to_visit)
        return nothing
    end

    return popfirst!(to_visit), to_visit
end

# For `collect` and list comprehensions on TreeTraversals to function
function Base.length(t::TreeTraversal)
    count = 0

    for _ in t
        count += 1
    end

    return count
end

# Implement AbstractTrees API

AbstractTrees.ChildIndexing(::Type{<:TreeNode}) = IndexedChildren()
AbstractTrees.children(node::TreeNode) = node.children
AbstractTrees.childrentype(::Type{<:TreeNode}) = Vector{TreeNode}
AbstractTrees.ParentLinks(::Type{<:TreeNode}) = StoredParents()
AbstractTrees.parent(node::TreeNode) = node.up
AbstractTrees.NodeType(::Type{<:TreeNode}) = HasNodeType()
AbstractTrees.nodetype(::Type{<:TreeNode}) = TreeNode
AbstractTrees.nodevalue(node::TreeNode) = node.type
Base.IteratorEltype(::Type{<:TreeIterator{TreeNode}}) = Base.HasEltype()
Base.eltype(::Type{<:TreeIterator{TreeNode}}) = TreeNode

Base.show(io::IO, node::TreeNode) = print(io, "TreeNode: $(node.event) event at time $(node.time) with type $(node.type)")


# To enable Plots.plot(tree::TreeNode).
# See ColorSchemes.colorschemes for all `colorscheme` options.
@recipe function _(tree::TreeNode; colorscheme=:linear_kbc_5_95_c73_n256)
    for node in PreOrderTraversal(tree)
        if length(node.children) > 2
            throw(ArgumentError("Only trees with at most binary branching are supported."))
        end
    end

    # Default series configuration
    xlabel --> "Time"
    yticks --> false
    yaxis --> false
    linewidth --> 1.5

    # First we must take note of the y-coordinate for each branch
    y_offsets = Dict{TreeNode, Float64}()

    # Set leaf y-coordinates
    for (i, node) in enumerate(LeafTraversal(tree))
        y_offsets[node] = i
    end

    # Determine internal node y-coordinates
    for node in PostOrderTraversal(tree)
        if node ∉ keys(y_offsets)
            y_offsets[node] = mean(y_offsets[child] for child in node.children)
        end
    end

    # Set up color palette. We only use the first 80% of the colorscheme because the last 20% are too light.
    all_types = sort(unique(node.type for node in PreOrderTraversal(tree)))
    num_colors = length(all_types)
    colors = (num_colors == 1) ? [colorschemes[colorscheme][1]] : colorschemes[colorscheme][0:0.8/(num_colors-1):0.8]
    palette = Dict(type => color for (type, color) in zip(all_types, colors))

    # Compute line segments from each node's parent to the node itself,
    # as well as connecting line segments for birth events.
    # Also compute segment colors
    for node in PreOrderTraversal(tree.children[1])
        @series begin
            x = [node.up.time, node.time]
            y = [y_offsets[node], y_offsets[node]]

            primary := false # Don't show up in legend
            seriescolor := palette[node.up.type]
            (x, y)
        end

        if node.event == :birth
            @series begin
                x = [node.time, node.time]
                y = [y_offsets[child] for child in node.children]

                primary := false # Don't show up in legend
                seriescolor := palette[node.up.type]
                (x, y)
            end
        end
    end

    # Create a dummy series to have a color legend
    for (type, color) in sort(palette)
        @series begin
            label := string(type)
            seriescolor := color
            seriestype := :shape
            ([], [])
        end
    end
end
