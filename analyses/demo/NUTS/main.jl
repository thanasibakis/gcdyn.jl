println("Loading packages...")

using gcdyn, CSV, DataFrames, Optim, Random, Turing

@model function Model(trees, Γ, type_space, present_time)
    # Keep priors on the same scale for NUTS
    log_λ_xscale ~ Normal(0, 1)
    λ_xshift⁻    ~ Normal(0, 1)
    log_λ_yscale ~ Normal(0, 1)
    log_λ_yshift ~ Normal(0, 1)
    log_μ        ~ Normal(0, 1)
    log_δ        ~ Normal(0, 1)

    # Obtain our actual parameters from the proxies
    λ_xscale = exp(log_λ_xscale * 0.75 + 0.5)
    λ_xshift = λ_xshift⁻ + 5
    λ_yscale = exp(log_λ_yscale * 0.75 + 0.5)
    λ_yshift = exp(log_λ_yshift * 1.2 - 0.5)
    μ        = exp(log_μ * 0.5)
    δ        = exp(log_δ * 0.5)

    if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
        sampled_model = VaryingTypeChangeRateBranchingProcess(
            λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, 1, 0, type_space, present_time
        )

        Turing.@addlogprob! loglikelihood(sampled_model, trees)
    end
end

function main()
    Random.seed!(1)

    println("Setting up model...")

    Γ = [-1 0.5 0.25 0.25; 2 -4 1 1; 2 2 -5 1; 0.125 0.125 0.25 -0.5]
    truth = VaryingTypeChangeRateBranchingProcess(1, 5, 1.5, 1, 1.3, 1, Γ, 1, 0, [2, 4, 6, 8], 3)
    trees = rand_tree(truth, 15, truth.type_space[1])

    model = Model(trees, truth.Γ, truth.type_space, truth.present_time)

    num_leaves = sort([length(LeafTraversal(tree)) for tree in trees])
    num_nodes = sort([length(PostOrderTraversal(tree)) for tree in trees])

    open("tree-summary.txt", "w") do file
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
