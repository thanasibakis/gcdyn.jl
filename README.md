# Multitype branching processes for inferring germinal center dynamics

## Setup

The `package` directory is a Julia package implementing the inferential models.
Scripts in the `analyses` directory assume this is installed as a package.
This is taken care of if you activate the Julia project at the root directory, using `julia --project`.

## Collection of self-contained analyses with this package

Note that each analysis in `analyses/` is self-contained, and similarly-named scripts across analyses may have differences.

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

This analysis requires the following setup:

- `pip install -e lib/gcdyn`
- Install R 4.2
- Install nextflow
