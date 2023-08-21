# Custom types.

"The possible events that can occur at a node in a [`TreeNode`](@ref)."
const EVENTS = (:root, :birth, :sampled_death, :unsampled_death, :mutation, :sampled_survival, :unsampled_survival)

"""
```julia
TreeNode(state)
TreeNode(event, t, state)
```
A data structure for building realizations of multitype branching processes, defaulting to time `t = 0` and the root `event`.

A "tree" is defined as a `TreeNode` with `event = :root` and optional `children`.
Defaults to having time `t` be `0` and event type be `:root`.
Requires an initial state value.

See also [`rand_tree`](@ref), [`EVENTS`](@ref).
"""
mutable struct TreeNode
    const event::Symbol
    const t::Float64
    state::Int
    const children::Vector{TreeNode}
    up::Union{TreeNode, Nothing}

    # A place to write temporary values pertaining to this node
    # (for calculations for the likelihood)
    p_start::Vector{Float64}
    p_end::Vector{Float64}
    q_start::Float64
    q_end::Float64

    function TreeNode(event, t, state)
        if event ∉ EVENTS
            throw(ArgumentError("Event must be one of $(EVENTS)"))
        elseif t < 0
            throw(ArgumentError("Time must be positive"))
        end

        return new(event, t, state, [], nothing, [], [], 0, 0)
    end
end

function TreeNode(state)
    return TreeNode(:root, 0, state)
end

"""
```julia
MultitypeBranchingProcess(λ, μ, γ, state_space, transition_matrix, ρ, σ, present_time)
```

Constructs a multitype branching process with the given parameters.

Rate parameters are notated as `λ` (birth rate), `μ` (death rate), and `γ` (mutation rate).
Sampling probabilities are notated as `ρ` (survival sampling probability) and `σ` (death sampling probability).
Rate parameters can be passed either as scalars or as functions that map state values to rates.
"""
struct MultitypeBranchingProcess
    λ::Function
    μ::Function
    γ::Function
    state_space::Vector{Int}
    transition_matrix::Matrix{Float64}
    ρ::Float64
    σ::Float64
    present_time::Float64

    function MultitypeBranchingProcess(λ::Union{Real, Function}, μ::Union{Real, Function}, γ::Union{Real, Function}, state_space, transition_matrix, ρ, σ, present_time)
        if ρ < 0 || ρ > 1
            throw(ArgumentError("ρ must be between 0 and 1"))
        elseif σ < 0 || σ > 1
            throw(ArgumentError("σ must be between 0 and 1"))
        elseif present_time < 0
            throw(ArgumentError("Time must be positive"))
        elseif length(state_space) != size(transition_matrix, 1)
            throw(DimensionMismatch("The number of states in the state space must match the number of rows in the transition matrix."))
        elseif length(state_space) != size(transition_matrix, 2)
            throw(DimensionMismatch("The number of states in the state space must match the number of columns in the transition matrix."))
        elseif any(transition_matrix .< 0) || any(transition_matrix .> 1)
            throw(ArgumentError("The transition matrix must contain only values between 0 and 1."))
        elseif any(sum(transition_matrix, dims=2) .!= 1)
            throw(ArgumentError("The transition matrix must contain only rows that sum to 1."))
        elseif any(!=(0), transition_matrix[i,i] for i in minimum(axes(transition_matrix))) && length(state_space) > 1
            throw(ArgumentError("The transition matrix must contain only zeros on the diagonal."))
        end

        # Convert scalars to constant functions
        λ⁺::Function = isa(λ, Function) ? λ : _ -> λ
        μ⁺::Function = isa(μ, Function) ? μ : _ -> μ
        γ⁺::Function = isa(γ, Function) ? γ : _ -> γ

        # Ensure functions return floating point numbers
        λ⁺⁺ = x -> convert(Float64, λ⁺(x))
        μ⁺⁺ = x -> convert(Float64, μ⁺(x))
        γ⁺⁺ = x -> convert(Float64, γ⁺(x))

        return new(λ⁺⁺, μ⁺⁺, γ⁺⁺, state_space, transition_matrix, ρ, σ, present_time)
    end
end

"""
```julia
MultitypeBranchingProcess(λ, μ, γ, state_space, ρ, σ, present_time)
```
Specifies uniform transition probabilities to all states.
"""
function MultitypeBranchingProcess(λ::Union{Real, Function}, μ::Union{Real, Function}, γ::Union{Real, Function}, state_space, ρ, σ, present_time)
    n = length(state_space)
    transition_matrix = (ones(n, n) - I) / (n - 1)

    return MultitypeBranchingProcess(λ, μ, γ, state_space, transition_matrix, ρ, σ, present_time)
end