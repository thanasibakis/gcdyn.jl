using gcdyn, StatsBase, Test
import Random

function test_rand_tree(n, λ, μ, present_time)
    # Assumes ρ = 1. σ should not matter

    model = ConstantBranchingProcess(λ, μ, 0, 1, 1, 1:3)
    trees = rand_tree(model, present_time, 1, n; reject_stubs=false)
    
    # Expected number of survivors should check out
    observed = map(trees) do tree
        sum([1 for node in LeafTraversal(tree) if node.event == :sampled_survival])
    end

    theoretical = exp(present_time * (λ - μ))

    @test mean(observed) ≈ theoretical rtol=1e-1
end

function test_fully_observed_likelihoods(n, λ, μ, γ, state_space, present_time)
    model = ConstantBranchingProcess(λ, μ, γ, 1, 1, state_space)
    trees = rand_tree(model, present_time, state_space[1], n)

    naive_ll = sum(gcdyn.naive_loglikelihood(model, tree) for tree in trees)
    appx_ll = sum(gcdyn.stadler_appx_loglikelihood(model, tree, present_time) for tree in trees)

    @test naive_ll ≈ loglikelihood(model, trees, present_time) rtol=1e-1
    @test naive_ll ≈ appx_ll                                   rtol=1e-1
end

function test_no_extinction_likelihoods(n, λ, μ, present_time)
    model = ConstantBranchingProcess(λ, μ, 0, 1, 0, 1:2)
    trees = rand_tree(model, present_time, 1, n)

    appx_ll = sum(gcdyn.stadler_appx_loglikelihood(model, tree, present_time) for tree in trees)

    @test appx_ll ≈ loglikelihood(model, trees, present_time) rtol=1e-1
end

@testset "gcdyn" begin
    Random.seed!(2)

    test_rand_tree(10000, 2.5, 1.1, 2)
    test_fully_observed_likelihoods(50, 2.5, 1.1, 1.1, 1:3, 3)
    test_no_extinction_likelihoods(50, 2.5, 1.1, 3)
end
