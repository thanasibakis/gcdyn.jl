# Multitype branching processes for inferring germinal center dynamics

## Setup

The `gcdyn.jl` directory is a Julia package implementing the inferential models.
Scripts in the `examples` directory assume this is installed as a package:

```
julia -e 'using Pkg; Pkg.develop(path="./gcdyn.jl")'
```

## Examples

### `examples/simulation-study`

Generates many sets of trees, performs Bayesian inference for each set, and examines the resulting sampling distribution of posterior medians across treesets.

### `examples/real-data`

Examines the effect of using a likelihood that does or does not condition on trees having at least one sampled descendant at present day.

### `examples/tenseqs`

Work in progress to run inference on BEAST trees.