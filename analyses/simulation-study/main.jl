println("Loading packages...")

using gcdyn, CSV, DataFrames, Random, Turing

@model function Model(trees, Γ, state_space, present_time)
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

    sampled_model = VaryingTypeChangeRateBranchingProcess(
        λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, 1, 0, state_space, present_time
    )

    if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
        Turing.@addlogprob! loglikelihood(sampled_model, trees)
    end
end

function main()
	println("Setting up model...")

    Γ = [-1 0.5 0.25 0.25; 2 -4 1 1; 2 2 -5 1; 0.125 0.125 0.25 -0.5]
    truth = VaryingTypeChangeRateBranchingProcess(1, 5, 1.5, 1, 1.3, 1, Γ, 1, 0, [2, 4, 6, 8], 3)

	println("Sampling from posterior...")
    num_treesets = 100
	num_trees_per_set = 15
    dfs = Vector{DataFrame}(undef, num_treesets)

    Threads.@threads for i in 1:num_treesets
        treeset = rand_tree(truth, num_trees_per_set, truth.state_space[1]);

        dfs[i] = sample(
            Model(treeset, truth.Γ, truth.state_space, truth.present_time),
            NUTS(),
            5000
        ) |> DataFrame

        dfs[i].run .= i
    end

    println("Exporting samples...")
    posterior_samples = vcat(dfs...)
    CSV.write("posterior-samples.csv", posterior_samples)

	println("Done!")
end

main()