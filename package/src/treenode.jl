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
Base.show(io::IO, ::MIME"text/plain", node::TreeNode) = show(io, MIME("text/plain"), D3Tree(node; detect_repeat=false))

"""
```julia
visualize_tree(io::IO=stdout, tree::TreeNode)
visualize_tree(path::String, tree::TreeNode)
```

Creates an HTML visualization of a `TreeNode` using D3Trees.
"""
function visualize_tree(io::IO, tree::TreeNode)
    d3 = D3Tree(tree; detect_repeat=false)

    # Color the tree
    all_types = sort(unique(node.type for node in PreOrderTraversal(tree)))
    scheme = map(ColorSchemes.tol_muted) do rgb
        "rgb($(round(Int, 255 * rgb.r)), $(round(Int, 255 * rgb.g)), $(round(Int, 255 * rgb.b)))"
    end

    if length(all_types) > length(scheme)
        throw(ArgumentError("Too many types to color"))
    end

    scheme_mapping = Dict(zip(all_types, scheme))

    # D3Trees stores this information in a vector ordered by a DFS
    style=["fill: $(scheme_mapping[node.type]); stroke: $(scheme_mapping[node.type]);" for node in PreOrderTraversal(tree)]
    link_style=["stroke: $(scheme_mapping[node.up.type]);" for node in PreOrderTraversal(tree) if !isnothing(node.up)]

    # Get tree depth
    paths = map(LeafTraversal(tree)) do leaf
        c = 0
        while !isnothing(leaf.up)
            c += 1
            leaf = leaf.up
        end
        c
    end
    depth = maximum(paths)

    d3 = D3Tree(d3, style=style, link_style=[""; link_style], init_expand=length(style), svg_height=120*depth)

    show(io, MIME("text/html"), d3)
end

visualize_tree(tree::TreeNode) = visualize_tree(stdout, tree)
visualize_tree(path::String, tree::TreeNode) = open(path, "w") do io
    visualize_tree(io, tree)
end