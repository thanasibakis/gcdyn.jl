# Core API

A [`gcdyn.TreeNode`](@ref) is the realization of a [`gcdyn.AbstractBranchingProcess`](@ref).

## Trees

```@docs
gcdyn.TreeNode

gcdyn.rand_tree

gcdyn.EVENTS

gcdyn.TreeTraversal
gcdyn.PostOrderTraversal
gcdyn.PreOrderTraversal
gcdyn.LeafTraversal

Plots.plot(::TreeNode)
```

## Branching processes

```@docs
gcdyn.AbstractBranchingProcess
gcdyn.ConstantBranchingProcess
gcdyn.DiscreteBranchingProcess
gcdyn.SigmoidalBranchingProcess

StatsAPI.loglikelihood

gcdyn.λ
gcdyn.μ
gcdyn.γ
```