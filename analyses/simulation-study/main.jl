println("Loading packages...")

using gcdyn, CSV, DataFrames, LinearAlgebra, Optim, Random, Turing

@model function Model(trees, Γ, type_space, present_time)
    # Keep priors on the same scale for NUTS
    θ ~ MvNormal(zeros(6), I)

    # Obtain our actual parameters from the proxies
    φ := [
        exp(θ[3] * 0.75 + 0.5),  # yscale
        exp(θ[1] * 0.75 + 0.5),  # xscale
        θ[2] + 5,                # xshift
        exp(θ[4] * 1.2 - 0.5),   # yshift
    ]
    μ := exp(θ[5] * 0.5)
    δ := exp(θ[6] * 0.5)

    if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
        sampled_model = SigmoidalBranchingProcess(φ[2], φ[3], φ[1], φ[4], μ, δ, Γ, 0.8, 0, type_space)

        Turing.@addlogprob! loglikelihood(sampled_model, trees, present_time)
    end
end

function main()
    Random.seed!(1)
    
	println("Setting up model...")

    type_space = [2, 4, 6, 8]
    Γ = [-1 0.5 0.25 0.25; 2 -4 1 1; 2 2 -5 1; 0.125 0.125 0.25 -0.5]
    present_time = 3
    truth = SigmoidalBranchingProcess(1, 5, 1.5, 1, 1.3, 1, Γ, 0.8, 0, type_space)

	println("Sampling from posterior...")
    num_treesets = 100
	num_trees_per_set = 15
    dfs = Vector{DataFrame}(undef, num_treesets)

    Threads.@threads for i in 1:num_treesets
        treeset = rand_tree(truth, present_time, truth.type_space[1], num_trees_per_set);
        model = Model(treeset, Γ, type_space, present_time)

        dfs[i] = sample(
            model,
            NUTS(adtype=AutoForwardDiff(chunksize=6)),
            1000,
            init_params=optimize(model, MAP(), NelderMead())
        ) |> DataFrame

        dfs[i].run .= i
    end

    println("Exporting samples...")
    posterior_samples = vcat(dfs...)
    CSV.write("posterior-samples.csv", posterior_samples)

	println("Done!")
end

main()
