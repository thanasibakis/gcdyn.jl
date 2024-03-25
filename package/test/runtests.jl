using gcdyn, StatsBase, Test

function test_rand_tree(n, λ, μ, present_time)
    # Assumes ρ = 1. σ should not matter

    model = FixedTypeChangeRateBranchingProcess(λ, μ, 0, 1, 1, 1:3, present_time)
    trees = rand_tree(model, n, 0; reject_stubs=false)
    
    # Expected number of survivors should check out
    observed = map(trees) do tree
        sum([1 for node in LeafTraversal(tree) if node.event == :sampled_survival])
    end

    theoretical = exp(present_time * (λ - μ))

    @test mean(observed) ≈ theoretical rtol=1e-1
end

function test_fully_observed_likelihoods(n, λ, μ, γ, state_space, present_time)
    model = FixedTypeChangeRateBranchingProcess(λ, μ, γ, 1, 1, state_space, present_time)
    trees = rand_tree(model, n, state_space[1])

    naive_ll = sum(gcdyn.naive_loglikelihood(model, tree) for tree in trees)
    appx_ll = sum(gcdyn.stadler_appx_loglikelihood(model, tree) for tree in trees)
    natural_ode_ll = sum(gcdyn.natural_ode_loglikelihood(model, tree) for tree in trees)

    @test naive_ll ≈ loglikelihood(model, trees) rtol=1e-1
    @test naive_ll ≈ natural_ode_ll              rtol=1e-1
    @test naive_ll ≈ appx_ll                     rtol=1e-1
end

function test_no_extinction_likelihoods(n, λ, μ, present_time)
    model = FixedTypeChangeRateBranchingProcess(λ, μ, 0, 1, 0, 1:2, present_time)
    trees = rand_tree(model, n, 1)

    appx_ll = sum(gcdyn.stadler_appx_loglikelihood(model, tree) for tree in trees)
    natural_ode_ll = sum(gcdyn.natural_ode_loglikelihood(model, tree) for tree in trees)

    @test appx_ll ≈ loglikelihood(model, trees) rtol=1e-1
    @test appx_ll ≈ natural_ode_ll              rtol=1e-1
end

function test_full_ode_likelihoods(n, λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, type_space, present_time)
    model = VaryingTypeChangeRateBranchingProcess(λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, 1, 0, type_space, present_time)
    trees = rand_tree(model, n, model.type_space[1])

    natural_ode_ll = sum(gcdyn.natural_ode_loglikelihood(model, tree) for tree in trees)

    @test loglikelihood(model, trees) ≈ natural_ode_ll rtol=1e-1
end

@testset "gcdyn" begin
    test_rand_tree(10000, 2.5, 1.1, 2)
    test_fully_observed_likelihoods(50, 2.5, 1.1, 1.1, 1:3, 3)
    test_no_extinction_likelihoods(50, 2.5, 1.1, 3)

    Γ = [-1 0.5 0.25 0.25; 2 -4 1 1; 2 2 -5 1; 0.125 0.125 0.25 -0.5]
    test_full_ode_likelihoods(50, 1, 5, 1.5, 1, 1.3, 1, Γ, [2, 4, 6, 8], 3)
end
