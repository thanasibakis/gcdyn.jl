module gcdyn

using
    AbstractTrees,
    DifferentialEquations,
    Distributions,
    LinearAlgebra,
    StatsAPI,
    StatsBase

export
    AbstractBranchingProcess,
    ConstantRateBranchingProcess,
    SigmoidalBirthRateBranchingProcess,
    TreeNode,
    uniform_transition_matrix,
    random_walk_transition_matrix,
    PostOrderTraversal,
    PreOrderTraversal,

    loglikelihood,
    rand_tree

include("types.jl")
include("treenode.jl")
include("multitype_branching_process.jl")
include("extra_likelihoods.jl")

end
