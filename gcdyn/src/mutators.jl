abstract type AbstractMutator end

struct DiscreteMutator <: AbstractMutator
    state_space::Vector{Real}
    transition_matrix::Matrix{Real}

    #function
end