using gcdyn, StatsBase, Test
import AbstractTrees

function test_rand_tree(n::Int, λ::Real, μ::Real, present_time::Real)
    # Assumes ρ = 1. σ should not matter

    trees = rand_tree(StadlerAppxModel(λ, μ, 1, 1, present_time), n; reject_stubs=false)
    
    # Expected number of survivors should check out
    observed = map(trees) do tree
        sum([1 for node in AbstractTrees.Leaves(tree) if node.event == :sampled_survival])
    end

    theoretical = exp(present_time * (λ - μ))

    @test mean(observed) ≈ theoretical atol=1
end

function test_fully_observed_likelihoods(n::Int, λ::Real, μ::Real, ρ::Real, present_time::Real)
    trees = rand_tree(NaiveModel(λ, μ, ρ, 1, present_time), n)

    @test loglikelihood(NaiveModel(λ, μ, ρ, 1, present_time), trees) ≈
          loglikelihood(StadlerAppxModel(λ, μ, ρ, 1, present_time), trees) atol=1e-3
end

@testset "gcdyn" begin
    test_rand_tree(10000, 2.5, 1.1, 2)
    test_fully_observed_likelihoods(1000, 2.5, 1.1, 1, 0.1)
end
