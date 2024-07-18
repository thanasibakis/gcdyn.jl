println("Loading packages...")
using gcdyn, CSV, DataFrames, LinearAlgebra, Optim, Random, Turing

@model function Model(trees, Γ, type_space, present_time)
    # Keep priors on the same scale for NUTS
    θ ~ MvNormal(zeros(6), I)

    # Obtain our actual parameters from the proxies
    λ_xscale := exp(θ[1] * 0.75 + 0.5)
    λ_xshift := θ[2] + 5
    λ_yscale := exp(θ[3] * 0.75 + 0.5)
    λ_yshift := exp(θ[4] * 1.2 - 0.5)
    μ        := exp(θ[5] * 0.5)
    δ        := exp(θ[6] * 0.5)

    if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
        sampled_model = SigmoidalBranchingProcess(
            λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, 1, 0, type_space
        )

        Turing.@addlogprob! loglikelihood(sampled_model, trees, present_time)
    end
end

function main()
    Random.seed!(1)

    println("Setting up model...")
    type_space = [2, 4, 6, 8]
    Γ = [-1 0.5 0.25 0.25; 2 -4 1 1; 2 2 -5 1; 0.125 0.125 0.25 -0.5]
    present_time = 3
    truth = SigmoidalBranchingProcess(1, 5, 1.5, 1, 1.3, 1, Γ, 1, 0, type_space)
    trees = rand_tree(truth, present_time, type_space[1], 15)

    model = Model(trees, Γ, type_space, present_time)

    open("tree-summary.txt", "w") do file
        num_leaves = sort([length(LeafTraversal(tree)) for tree in trees])
        num_nodes = sort([length(PostOrderTraversal(tree)) for tree in trees])

        println(file, "Leaf count distribution:")
        println(file, join(num_leaves, ", "), "\n")
        println(file, "Node count distribution:")
        println(file, join(num_nodes, ", "))
    end

    println("Computing initial MCMC state...")
    max_a_posteriori = optimize(model, MAP(), NelderMead())

    println("Sampling from posterior...")
    posterior_samples = sample(model, NUTS(adtype=AutoForwardDiff(chunksize=6)), 1000, init_params=max_a_posteriori) |> DataFrame

    println("Exporting samples...")
    CSV.write("posterior-samples.csv", posterior_samples)

    print("Done!")    
end

main()
