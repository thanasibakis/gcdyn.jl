module gcdyn

using
    AbstractTrees,
    DataFrames,
    Distributions,
    LinearAlgebra,
    OrdinaryDiffEq,
    StatsAPI,
    StatsBase

export
    AbstractBranchingProcess,
    FixedTypeChangeRateBranchingProcess,
    VaryingTypeChangeRateBranchingProcess,
    TreeNode,
    uniform_transition_matrix,
    random_walk_transition_matrix,
    discretize_states!,
    LeafTraversal,
    PostOrderTraversal,
    PreOrderTraversal,
    λ,
    μ,
    γ,
    loglikelihood,
    rand_tree,
    map_states!

include("types.jl")
include("treenode.jl")
include("multitype_branching_process.jl")
include("extra_likelihoods.jl")

end
