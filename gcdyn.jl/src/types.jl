# Custom types and their constructors.
# Note: since Julia expects types to be defined before any functions that use them,
# this file should contain only types and be `include`d before other files.

"The possible events that can occur at a node in a [`TreeNode`](@ref)."
const EVENTS = (:root, :birth, :sampled_death, :unsampled_death, :mutation, :sampled_survival, :unsampled_survival)

"""
```julia
TreeNode(state)
TreeNode(event, t, state)
TreeNode(name, event, t, state, children)
```
A data structure for building realizations of multitype branching processes.

A "tree" is defined as a `TreeNode` with `event = :root` and optional `TreeNode` `children` of other non-root events.
All nodes in the tree (including the root) have an integer `name`, a time `t` at which the event occurred, and a `state` attribute whose meaning is left to the user.

If only `state` is provided, the node defaults to being a root node at time 0.
Node names default to `0` for root nodes, and `1` for all others.
Child vectors always default to be empty.

See also [`rand_tree`](@ref), [`EVENTS`](@ref).
"""
mutable struct TreeNode
    name::Int
    event::Symbol
    t::Float64
    state::Float64
    const children::Vector{TreeNode}
    up::Union{TreeNode, Nothing}

    p_start::Vector{Float64}
    p_end::Vector{Float64}
    q_start::Float64
    q_end::Float64

    # A not-so-type-stable way to store any extra info
    info::Dict

    function TreeNode(name, event, t, state, children)
        if event ∉ EVENTS
            throw(ArgumentError("Event must be one of $(EVENTS)"))
        elseif t < 0
            throw(ArgumentError("Time must be positive"))
        end

        self = new(name, event, t, state, children, nothing, [], [], 0, 0, Dict())

        for child in self.children
            child.up = self
        end

        return self
    end
end

TreeNode(state) = TreeNode(0, :root, 0, state, [])
TreeNode(event, t, state) = TreeNode(event == :root ? 0 : 1, event, t, state, [])

"""
```julia
AbstractBranchingProcess
```
Abstract supertype for branching processes.

See also [`ConstantRateBranchingProcess`](@ref), [`SigmoidalBirthRateBranchingProcess`](@ref).
"""
abstract type AbstractBranchingProcess end

"""
```julia
ConstantRateBranchingProcess(λ, μ, γ, state_space, [transition_matrix,] ρ, σ, present_time)
```

Constructs a multitype branching process with the given parameters.

Rate parameters are notated as `λ` (birth rate), `μ` (death rate), and `γ` (mutation rate).
Sampling probabilities are notated as `ρ` (survival sampling probability) and `σ` (death sampling probability).
Ensure that `present_time > 0`, since the process is defined to start at time `0`.

If `transition_matrix` is omitted, specifies uniform transition probabilities to all states.

See also [`uniform_transition_matrix`](@ref), [`random_walk_transition_matrix`](@ref).
"""
struct ConstantRateBranchingProcess <: AbstractBranchingProcess
    λ::Float64
    μ::Float64
    γ::Float64
    state_space::Vector{Int}
    transition_matrix::Matrix{Float64}
    ρ::Float64
    σ::Float64
    present_time::Float64

    function ConstantRateBranchingProcess(λ, μ, γ, state_space, transition_matrix, ρ, σ, present_time)
        if λ < 0 || μ < 0 || γ < 0
            throw(ArgumentError("Rate parameters must be positive"))
        elseif ρ < 0 || ρ > 1
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

        return new(λ, μ, γ, state_space, transition_matrix, ρ, σ, present_time)
    end
end

function ConstantRateBranchingProcess(λ, μ, γ, state_space, ρ, σ, present_time)
    n = length(state_space)
    transition_matrix = (ones(n, n) - I) / (n - 1)

    return ConstantRateBranchingProcess(λ, μ, γ, state_space, transition_matrix, ρ, σ, present_time)
end

"""
```julia
SigmoidalBirthRateBranchingProcess(xscale, xshift, yscale, yshift, μ, γ, state_space, [transition_matrix,] ρ, σ, present_time)
```

Constructs a multitype branching process with the given parameters.

A parameterized sigmoid curve (with `xscale`, `xshift`, `yscale`, and `yshift`) maps states to birth rates.
Other rate parameters are notated as `μ` (death rate), and `γ` (mutation rate).
Sampling probabilities are notated as `ρ` (survival sampling probability) and `σ` (death sampling probability).
Ensure that `present_time > 0`, since the process is defined to start at time `0`.

If `transition_matrix` is omitted, specifies uniform transition probabilities to all states.

See also [`uniform_transition_matrix`](@ref), [`random_walk_transition_matrix`](@ref).
"""
struct SigmoidalBirthRateBranchingProcess <: AbstractBranchingProcess
    xscale::Float64
    xshift::Float64
    yscale::Float64
    yshift::Float64
    μ::Float64
    γ::Float64
    state_space::Vector{Int}
    transition_matrix::Matrix{Float64}
    ρ::Float64
    σ::Float64
    present_time::Float64

    function SigmoidalBirthRateBranchingProcess(xscale, xshift, yscale, yshift, μ, γ, state_space, transition_matrix, ρ, σ, present_time)
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

        return new(xscale, xshift, yscale, yshift, μ, γ, state_space, transition_matrix, ρ, σ, present_time)
    end
end

function SigmoidalBirthRateBranchingProcess(xscale, xshift, yscale, yshift, μ, γ, state_space, ρ, σ, present_time)
    transition_matrix = uniform_transition_matrix(state_space)

    return SigmoidalBirthRateBranchingProcess(xscale, xshift, yscale, yshift, μ, γ, state_space, transition_matrix, ρ, σ, present_time)
end

"""
```julia
uniform_transition_matrix(state_space)
```

Constructs a transition matrix where all states have equal probability of transitioning to all other states.

Note that self-loops cannot occur (ie. the diagonal of the matrix is all zeros).
"""
function uniform_transition_matrix(state_space)
    n = length(state_space)

    return (ones(n, n) - I) / (n - 1)
end

"""
```julia
random_walk_transition_matrix(state_space, p; δ=1)
```

Constructs a transition matrix where the probability of transitioning to the next state is `p`, and the probability of transitioning to the previous state is `1 - p`.

Additionally, if a scaling parameter δ is applied, then the transition probability `p` is scaled by `δⁱ`, where `i` is the number of transitions needed to reach the target state from
the first state in `state_space`.
"""
function random_walk_transition_matrix(state_space, p; δ=1)
    if p <= 0 || p >= 1
        throw(ArgumentError("p must be between 0 and 1"))
    end

    n = length(state_space)
    transition_matrix = zeros(n, n)

    for i in 1:n
        if i == 1
            transition_matrix[i, i+1] = 1
        elseif i == n
            transition_matrix[i, i-1] = 1
        else
            transition_matrix[i, i+1] = p * δ^(i-1)
            transition_matrix[i, i-1] = 1 - p * δ^(i-1)
        end
    end

    return transition_matrix
end
