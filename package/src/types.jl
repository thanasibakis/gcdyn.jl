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
An abstract type for branching processes.

See also [`SigmoidalBranchingProcess`](@ref), [`ConstantBranchingProcess`](@ref), [`DiscreteBranchingProcess`](@ref).
"""
abstract type AbstractBranchingProcess{T <: Real} end

# To allow us to broadcast the rate parameter functions over types
Base.broadcastable(model::AbstractBranchingProcess) = Ref(model)

"""
```julia
SigmoidalBranchingProcess(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, γ, ρ, σ, type_space, present_time)
SigmoidalBranchingProcess(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, ρ, σ, type_space, present_time)
```

Constructs a multitype branching process with the given parameters.

A parameterized sigmoid curve (with `λ_xscale`, `λ_xshift`, `λ_yscale`, and `λ_yshift`) maps types to birth rates.
The death rate parameter is notated as `μ`.

`Γ` is the type change rate matrix, and `δ` is a scaling parameter for this matrix.
Alternatively, a single type change rate `γ` can be provided for all types, in which case the model will assume uniform transitions between types.

Sampling probabilities are notated as `ρ` (survival sampling probability) and `σ` (death sampling probability).
Ensure that `present_time > 0`, since the process is defined to start at time `0`.
"""
struct SigmoidalBranchingProcess{T <: Real} <: AbstractBranchingProcess{T}
    λ_xscale::T
    λ_xshift::T
    λ_yscale::T
    λ_yshift::T
    μ::T
    δ::T
    Γ::Matrix{Float64}
    ρ::Float64
    σ::Float64
    type_space::Vector{Float64}
    present_time::Float64

    function SigmoidalBranchingProcess(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, ρ, σ, type_space, present_time)
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

        # Types are parameterized in case autodiff is used. But if it's an Int, we really want a Float
        return new{typeof(λ_xscale*1.0)}(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, ρ, σ, type_space, present_time)
    end
end

function SigmoidalBranchingProcess(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, γ, ρ, σ, type_space, present_time)
    n = length(type_space)
    Γ = ones(n, n) / (n-1) * γ
    Γ[diagind(Γ)] .= -γ
    
    return SigmoidalBranchingProcess(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, 1, Γ, ρ, σ, type_space, present_time)
end

"""
```julia
ConstantBranchingProcess(λ, μ, γ, ρ, σ, type_space, present_time)
ConstantBranchingProcess(λ, μ, δ, Γ, ρ, σ, type_space, present_time)
```

Constructs a multitype branching process with the given parameters.

A single birth rate `λ` is provided for all types.
The death rate parameter is notated as `μ`.

`Γ` is the type change rate matrix, and `δ` is a scaling parameter for this matrix.
Alternatively, a single type change rate `γ` can be provided for all types, in which case the model will assume uniform transitions between types.

Sampling probabilities are notated as `ρ` (survival sampling probability) and `σ` (death sampling probability).
Ensure that `present_time > 0`, since the process is defined to start at time `0`.

"""
struct ConstantBranchingProcess{T <: Real} <: AbstractBranchingProcess{T}
    λ::T
    μ::T
    δ::T
    Γ::Matrix{Float64}
    ρ::Float64
    σ::Float64
    type_space::Vector{Float64}
    present_time::Float64

    function ConstantBranchingProcess(λ, μ, δ, Γ, ρ, σ, type_space, present_time)
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

        return new{typeof(λ*1.0)}(λ, μ, δ, Γ, ρ, σ, type_space, present_time)
    end
end

function ConstantBranchingProcess(λ, μ, γ, ρ, σ, type_space, present_time)
    n = length(type_space)
    Γ = ones(n, n) / (n-1) * γ
    Γ[diagind(Γ)] .= -γ

    return ConstantBranchingProcess(λ, μ, 1, Γ, ρ, σ, type_space, present_time)
end

"""
```julia
DiscreteBranchingProcess(λ, μ, γ, ρ, σ, type_space, present_time)
DiscreteBranchingProcess(λ, μ, δ, Γ, ρ, σ, type_space, present_time)
```

Constructs a multitype branching process with the given parameters.

A `Vector` of birth rates `λ` is provided, with indices corresponding to `type_space`.
The death rate parameter is notated as `μ`.

`Γ` is the type change rate matrix, and `δ` is a scaling parameter for this matrix.
Alternatively, a single type change rate `γ` can be provided for all types, in which case the model will assume uniform transitions between types.

Sampling probabilities are notated as `ρ` (survival sampling probability) and `σ` (death sampling probability).
Ensure that `present_time > 0`, since the process is defined to start at time `0`.

"""
struct DiscreteBranchingProcess{T <: Real} <: AbstractBranchingProcess{T}
    λ::Vector{T}
    μ::T
    δ::T
    Γ::Matrix{Float64}
    ρ::Float64
    σ::Float64
    type_space::Vector{Float64}
    present_time::Float64

    function DiscreteBranchingProcess(λ, μ, δ, Γ, ρ, σ, type_space, present_time)
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

        return new{typeof(λ[1]*1.0)}(λ, μ, δ, Γ, ρ, σ, type_space, present_time)
    end
end

function DiscreteBranchingProcess(λ, μ, γ, ρ, σ, type_space, present_time)
    n = length(type_space)
    Γ = ones(n, n) / (n-1) * γ
    Γ[diagind(Γ)] .= -γ

    return DiscreteBranchingProcess(λ, μ, 1, Γ, ρ, σ, type_space, present_time)
end