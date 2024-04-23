module gcdyn

using
    AbstractTrees,
    ColorSchemes,
    DataFrames,
    Distributions,
    LinearAlgebra,
    Memoize,
    OrdinaryDiffEq,
    Plots,
    StatsAPI,
    StatsBase

export
    ConstantBranchingProcess,
    DiscreteBranchingProcess,
    SigmoidalBranchingProcess,
    TreeNode,
    LeafTraversal,
    PostOrderTraversal,
    PreOrderTraversal,
    λ,
    μ,
    γ,
    loglikelihood,
    rand_tree,
    map_types!,
    plot

include("types.jl")
include("treenode.jl")
include("multitype_branching_process.jl")
include("extra_likelihoods.jl")

end
