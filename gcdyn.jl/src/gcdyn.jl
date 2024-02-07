module gcdyn

using
    AbstractTrees,
    CategoricalArrays,
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

    loglikelihood,
    rand_tree

include("types.jl")
include("treenode.jl")
include("multitype_branching_process.jl")
include("extra_likelihoods.jl")

end
