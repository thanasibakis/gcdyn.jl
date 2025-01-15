# Other

## Trees

```@docs
gcdyn.mutate!
gcdyn.sample_child!

gcdyn.attach!
gcdyn.detach!
Base.delete!(::TreeNode)

gcdyn.map_types
gcdyn.map_types!
```

## Branching processes

```@docs
gcdyn.expit
gcdyn.sigmoid

gcdyn.dp_logq_dt!
gcdyn.dp_dt!

gcdyn.type_space_index
```

### Additional likelihood functions

```@docs
gcdyn.naive_loglikelihood
gcdyn.stadler_appx_loglikelihood
gcdyn.stadler_appx_unconditioned_loglikelihood
```