using gcdyn, AbstractTrees, StatsBase, Test

function test_rand_tree(n, λ, μ, present_time)
    # Assumes ρ = 1. σ should not matter

    model = FixedTypeChangeRateBranchingProcess(λ, μ, 0, 1, 1, 1:3, present_time)
    trees = rand_tree(model, n, 0; reject_stubs=false)
    
    # Expected number of survivors should check out
    observed = map(trees) do tree
        sum([1 for node in LeafTraversal(tree) if node.event == :sampled_survival])
    end

    theoretical = exp(present_time * (λ - μ))

    @test mean(observed) ≈ theoretical atol=1
end

function test_fully_observed_likelihoods(n, λ, μ, γ, state_space, present_time)
    model = FixedTypeChangeRateBranchingProcess(λ, μ, γ, 1, 1, state_space, present_time)
    trees = rand_tree(model, n, state_space[1])

    naive_ll = sum(gcdyn.naive_loglikelihood(model, tree) for tree in trees)
    appx_ll = sum(gcdyn.stadler_appx_loglikelhood(model, tree) for tree in trees)

    @test loglikelihood(model, trees) ≈ naive_ll atol=1e-1
    @test naive_ll ≈ appx_ll atol=1e-1
end

function test_stadler_likelihoods(n, λ, μ, present_time)
    model = FixedTypeChangeRateBranchingProcess(λ, μ, 0, 1, 0, 1:2, present_time)
    trees = rand_tree(model, n, 1)

    appx_ll = sum(gcdyn.stadler_appx_loglikelhood(model, tree) for tree in trees)

    @test loglikelihood(model, trees) ≈ appx_ll atol=1e-1
end

@testset "gcdyn" begin
    test_rand_tree(10000, 2.5, 1.1, 2)
    test_fully_observed_likelihoods(1000, 2.5, 1.1, 0, 1:3, 1)
    test_stadler_likelihoods(1000, 2.5, 1.1, 1)
end
