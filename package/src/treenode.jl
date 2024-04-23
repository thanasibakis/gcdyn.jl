# Methods involving `TreeNode` objects.

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
@recipe function _(tree::TreeNode; colorscheme=:mk_15)
    # Default series configuration
    xlabel --> "Time"
    yticks --> false
    yaxis --> false

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

    # Set up color palette
    all_types = sort(unique(node.type for node in PreOrderTraversal(tree)))
    num_colors = length(all_types)
    colors = (num_colors == 1) ? colorschemes[colorscheme][0] : colorschemes[colorscheme][0:1/(num_colors-1):1]
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
            label := round(type; digits=3) |> string
            seriescolor := color
            seriestype := :shape
            ([], [])
        end
    end
end
