# Multitype branching processes for inferring germinal center dynamics

## Setup

```
# Install matsengrp/gcdyn Python package, which has tree samplers
pip install -e ./gcdyn/gcdyn

# Install Julia package implementing the inferential models
julia -e 'using Pkg; Pkg.develop(path="./gcdyn.jl")'
```
