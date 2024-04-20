module gcdyn

using
    AbstractTrees,
    ColorSchemes,
    DataFrames,
    Distributions,
    D3Trees,
    LinearAlgebra,
    Memoize,
    OrdinaryDiffEq,
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
    visualize_tree,
    map_types!

include("types.jl")
include("treenode.jl")
include("multitype_branching_process.jl")
include("extra_likelihoods.jl")

end
