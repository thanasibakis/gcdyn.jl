module gcdyn

using
    AbstractTrees,
    DifferentialEquations,
    Distributions,
    LinearAlgebra,
    StatsAPI,
    StatsBase

export
    MultitypeBranchingProcess,
    TreeNode,

    loglikelihood,
    rand_tree

include("types.jl")
include("treenode.jl")
include("multitype_branching_process.jl")
include("extra_likelihoods.jl")

end
