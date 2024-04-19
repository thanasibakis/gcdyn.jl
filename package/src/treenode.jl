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
