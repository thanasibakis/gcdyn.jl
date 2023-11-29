# Collection of self-contained example usages of `gcdyn.jl`

All Julia scripts in these examples should be run with `julia --project [script]` to activate the environment stored in this directory.

Note that each example is self-contained, and similarly-named scripts across examples may have differences.

## demo

A simple demo of inference on a single set of trees simulated from the branching process.

## simulation-study

A systematic study of the branching process, conducting inference on many sets of simulated trees and understanding the model's frequentist properties.

## simulation-study-misspecification

Similar to `simulation-study`, but instead of simulating trees from the branching process, they are simulated according to mutations on the genotype and subject to a carrying capacity.

## error-analysis

An exploration of how inference can be biased if we exclude extinct trees from our datasets but do not condition on non-extinction in the model.

## beast-analysis

Inference conducted on trees inferred from real sequence data using BEAST.

This example requires the following setup:

```
pip install -e lib/gcdyn
``` 
