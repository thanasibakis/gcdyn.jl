using gcdyn, StatsBase, Test
import AbstractTrees

function test_rand_tree(n::Int, λ::Real, μ::Real, present_time::Real)
    # Assumes ρ = 1. σ should not matter

    trees = rand_tree(StadlerAppxModel(_ -> λ, _ -> μ, _ -> 0, DiscreteMutator(1:3), 1, 1, present_time), n, 0; reject_stubs=false)
    
    # Expected number of survivors should check out
    observed = map(trees) do tree
        sum([1 for node in AbstractTrees.Leaves(tree) if node.event == :sampled_survival])
    end

    theoretical = exp(present_time * (λ - μ))

    @test mean(observed) ≈ theoretical atol=1
end

function test_fully_observed_likelihoods(n::Int, λ::Function, μ::Function, γ::Function, mutator::AbstractMutator, present_time::Real)
    trees = rand_tree(NaiveModel(λ, μ, γ, mutator, 1, 1, present_time), n, mutator.state_space[1])

    @test loglikelihood(NaiveModel(λ, μ, γ, mutator, 1, 1, present_time), trees) ≈
          loglikelihood(StadlerAppxModel(λ, μ, γ, mutator, 1, 1, present_time), trees) atol=1e-3
end

@testset "gcdyn" begin
    test_rand_tree(10000, 2.5, 1.1, 2)
    test_fully_observed_likelihoods(1000, _ -> 2.5, _ -> 1.1, _ -> 0, DiscreteMutator(1:3), 0.1)
    test_fully_observed_likelihoods(1000, x -> x, _ -> 1.1, _ -> 2, DiscreteMutator(1:3), 0.1)
end
