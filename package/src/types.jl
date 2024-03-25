# Custom types and their constructors.
# Note: since Julia expects types to be defined before any functions that use them,
# this file should contain only types and be `include`d before other files.

"The possible events that can occur at a node in a [`TreeNode`](@ref)."
const EVENTS = (:root, :birth, :sampled_death, :unsampled_death, :type_change, :sampled_survival, :unsampled_survival)

"""
```julia
TreeNode(type)
TreeNode(event, time, type)
TreeNode(name, event, time, type, children)
```
A data structure for building realizations of multitype branching processes.

A "tree" is defined as a `TreeNode` with `event = :root` and optional `TreeNode` `children` of other non-root events.
All nodes in the tree (including the root) have an integer `name`, a `time` at which the event occurred, and a `type` attribute whose meaning is left to the user.

If only `type` is provided, the node defaults to being a root node at time 0.
Node names default to `0` for root nodes, and `1` for all others.
Child vectors always default to be empty.

See also [`rand_tree`](@ref), [`EVENTS`](@ref).
"""
mutable struct TreeNode
    name::Int
    event::Symbol
    time::Float64
    type::Float64
    const children::Vector{TreeNode}
    up::Union{TreeNode, Nothing}

    # A not-so-type-stable way to store any extra info
    info::Dict

    function TreeNode(name, event, time, type, children)
        if event ∉ EVENTS
            throw(ArgumentError("Event must be one of $(EVENTS)"))
        elseif time < 0
            throw(ArgumentError("Time must be positive"))
        end

        self = new(name, event, time, type, children, nothing, Dict())

        for child in self.children
            child.up = self
        end

        return self
    end
end

TreeNode(type) = TreeNode(0, :root, 0, type, [])
TreeNode(event, time, type) = TreeNode(event == :root ? 0 : 1, event, time, type, [])

"""
```julia
AbstractBranchingProcess
```
Abstract supertype for branching processes.

See also [`FixedTypeChangeRateBranchingProcess`](@ref), [`VaryingTypeChangeRateBranchingProcess`](@ref).
"""
abstract type AbstractBranchingProcess end

"""
```julia
FixedTypeChangeRateBranchingProcess(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, γ, [Π,] ρ, σ, type_space, present_time)
```

Constructs a multitype branching process with the given parameters.

A parameterized sigmoid curve (with `λ_xscale`, `λ_xshift`, `λ_yscale`, and `λ_yshift`) maps types to birth rates.
Other rate parameters are notated as `μ` (death rate), and `γ` (type change rate).
Sampling probabilities are notated as `ρ` (survival sampling probability) and `σ` (death sampling probability).
Ensure that `present_time > 0`, since the process is defined to start at time `0`.

If a transition matrix `Π` is omitted, specifies uniform transition probabilities to all types.

See also [`uniform_transition_matrix`](@ref), [`random_walk_transition_matrix`](@ref).
"""
struct FixedTypeChangeRateBranchingProcess{R₁ <: Real, R₂ <: Real, R₃ <: Real, R₄ <: Real, R₅ <: Real, R₆ <: Real, M <: AbstractMatrix{R} where R <: Real, S <: AbstractVector{T} where T <: Real} <: AbstractBranchingProcess
    λ_xscale::R₁
    λ_xshift::R₂
    λ_yscale::R₃
    λ_yshift::R₄
    μ::R₅
    γ::R₆
    Π::M
    ρ::Float64
    σ::Float64
    type_space::S
    present_time::Float64

    function FixedTypeChangeRateBranchingProcess(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, γ, Π, ρ, σ, type_space, present_time)
        if ρ < 0 || ρ > 1
            throw(ArgumentError("ρ must be between 0 and 1"))
        elseif σ < 0 || σ > 1
            throw(ArgumentError("σ must be between 0 and 1"))
        elseif present_time < 0
            throw(ArgumentError("Time must be positive"))
        elseif length(type_space) != size(Π, 1)
            throw(DimensionMismatch("The number of types in the type space must match the number of rows in the transition matrix."))
        elseif length(type_space) != size(Π, 2)
            throw(DimensionMismatch("The number of types in the type space must match the number of columns in the transition matrix."))
        elseif any(Π .< 0) || any(Π .> 1)
            throw(ArgumentError("The transition matrix must contain only values between 0 and 1."))
        elseif any(≉(1; atol=1e-10), sum(Π, dims=2))
           throw(ArgumentError("The transition probability matrix must contain only rows that sum to 1."))
        elseif any(!=(0), Π[i,i] for i in minimum(axes(Π))) && length(type_space) > 1
           throw(ArgumentError("The transition probability matrix must contain only zeros on the diagonal."))
        end

        return new{typeof(λ_xscale), typeof(λ_xshift), typeof(λ_yscale), typeof(λ_yshift), typeof(μ), typeof(γ), typeof(Π), typeof(type_space),}(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, γ, Π, ρ, σ, type_space, present_time)
    end
end

function FixedTypeChangeRateBranchingProcess(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, γ, ρ, σ, type_space, present_time)
    Π = uniform_transition_matrix(type_space)

    return FixedTypeChangeRateBranchingProcess(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, γ, Π, ρ, σ, type_space, present_time)
end

"""
```julia
FixedTypeChangeRateBranchingProcess(λ, μ, γ, [transition_matrix,] ρ, σ, type_space, present_time)
```

Constructs a multitype branching process with the given parameters.

Rate parameters are notated as `λ` (birth rate), `μ` (death rate), and `γ` (type change rate).
Sampling probabilities are notated as `ρ` (survival sampling probability) and `σ` (death sampling probability).
Ensure that `present_time > 0`, since the process is defined to start at time `0`.

If a transition matrix `Π` is omitted, specifies uniform transition probabilities to all types.

See also [`uniform_transition_matrix`](@ref), [`random_walk_transition_matrix`](@ref).
"""
function FixedTypeChangeRateBranchingProcess(λ, μ, γ, Π, ρ, σ, type_space, present_time)
    return FixedTypeChangeRateBranchingProcess(0, 0, 0, λ, μ, γ, Π, ρ, σ, type_space, present_time)
end

function FixedTypeChangeRateBranchingProcess(λ, μ, γ, ρ, σ, type_space, present_time)
    Π = uniform_transition_matrix(type_space)

    return FixedTypeChangeRateBranchingProcess(λ, μ, γ, Π, ρ, σ, type_space, present_time)
end

"""
```julia
VaryingTypeChangeRateBranchingProcess(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, ρ, σ, type_space, present_time)
```

Constructs a multitype branching process with the given parameters.

Similar to [`FixedTypeChangeRateBranchingProcess`](@ref), but with a reparameterization regarding type changes.
`δ` is a scaling parameter for the known type change rate matrix `Γ`, which replaces the constant type change rate and transition probability matrix.
"""
struct VaryingTypeChangeRateBranchingProcess{R₁ <: Real, R₂ <: Real, R₃ <: Real, R₄ <: Real, R₅ <: Real, R₆ <: Real, M <: AbstractMatrix{R} where R <: Real, S <: AbstractVector{T} where T <: Real} <: AbstractBranchingProcess
    λ_xscale::R₁
    λ_xshift::R₂
    λ_yscale::R₃
    λ_yshift::R₄
    μ::R₅
    δ::R₆
    Γ::M
    ρ::Float64
    σ::Float64
    type_space::S
    present_time::Float64

    function VaryingTypeChangeRateBranchingProcess(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, ρ, σ, type_space, present_time)
        if ρ < 0 || ρ > 1
            throw(ArgumentError("ρ must be between 0 and 1"))
        elseif σ < 0 || σ > 1
            throw(ArgumentError("σ must be between 0 and 1"))
        elseif present_time < 0
            throw(ArgumentError("Time must be positive"))
        elseif length(type_space) != size(Γ, 1)
            throw(DimensionMismatch("The number of types in the type space must match the number of rows in the rate matrix."))
        elseif length(type_space) != size(Γ, 2)
            throw(DimensionMismatch("The number of types in the type space must match the number of columns in the rate matrix."))
        elseif any(≉(0; atol=1e-10), sum(Γ, dims=2))
           throw(ArgumentError("The transition rate matrix must contain only rows that sum to 0."))
        elseif any(>(0), Γ[i,i] for i in minimum(axes(Γ))) && length(type_space) > 1
           throw(ArgumentError("The transition rate matrix must contain only negative or zero values on the diagonal."))
        end

        return new{typeof(λ_xscale), typeof(λ_xshift), typeof(λ_yscale), typeof(λ_yshift), typeof(μ), typeof(δ), typeof(Γ), typeof(type_space)}(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, ρ, σ, type_space, present_time)
    end
end

"""
```julia
uniform_transition_matrix(type_space)
```

Constructs a transition matrix where all types have equal probability of transitioning to all other types.

Note that self-loops cannot occur (ie. the diagonal of the matrix is all zeros).
"""
function uniform_transition_matrix(type_space)
    n = length(type_space)

    return (ones(n, n) - I) / (n - 1)
end

"""
```julia
random_walk_transition_matrix(type_space, p; δ=1)
```

Constructs a transition matrix where the probability of transitioning to the next type is `p`, and the probability of transitioning to the previous type is `1 - p`.

Additionally, if a scaling parameter δ is applied, then the transition probability `p` is scaled by `δⁱ`, where `i` is the number of transitions needed to reach the target type from
the first type in `type_space`.
"""
function random_walk_transition_matrix(type_space, p; δ=1)
    if p <= 0 || p >= 1
        throw(ArgumentError("p must be between 0 and 1"))
    end

    n = length(type_space)
    transition_matrix = zeros(typeof(p), n, n)

    for i in 1:n
        if i == 1
            transition_matrix[i, i+1] = 1
        elseif i == n
            transition_matrix[i, i-1] = 1
        else
            transition_matrix[i, i+1] = p * δ^(i-2)
            transition_matrix[i, i-1] = 1 - p * δ^(i-2)
        end
    end

    return transition_matrix
end
