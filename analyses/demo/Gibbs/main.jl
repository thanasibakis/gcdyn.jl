println("Loading packages...")

using gcdyn, CSV, DataFrames, Turing, StatsPlots

@model function Model(trees, Γ, type_space, present_time)
    λ_xscale  ~ Gamma(2, 1)
    λ_xshift  ~ Normal(5, 1)
    λ_yscale  ~ Gamma(2, 1)
    λ_yshift  ~ Gamma(1, 1)
    μ       ~ LogNormal(0, 0.5)
    δ       ~ LogNormal(0, 0.5)

    if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
        sampled_model = VaryingTypeChangeRateBranchingProcess(
            λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, 1, 0, type_space, present_time
        )

        Turing.@addlogprob! loglikelihood(sampled_model, trees)
    end
end

function main()
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

    println("Sampling from prior...")
    prior_samples = sample(model, Prior(), 100) |> DataFrame

    println("Sampling from posterior...")
    posterior_samples = sample(
        model,
        Gibbs(
            MH(:λ_xscale => x -> LogNormal(log(x), 0.2)),
            MH(:λ_xshift => x -> Normal(x, 0.7)),
            MH(:λ_yscale => x -> LogNormal(log(x), 0.2)),
            MH(:λ_yshift => x -> LogNormal(log(x), 0.2)),
            MH(:μ => x -> LogNormal(log(x), 0.2)),
            MH(:δ => x -> LogNormal(log(x), 0.2)),
        ),
        5000
    ) |> DataFrame

    println("Exporting samples...")
    CSV.write("posterior-samples.csv", posterior_samples)

    println("Visualizing...")

    plot(xlims=(0, 10), ylims=(0, 6), dpi=300)

    for row in eachrow(prior_samples)
        plot!(x -> gcdyn.sigmoid(x, row.λ_xscale, row.λ_xshift, row.λ_yscale, row.λ_yshift); alpha=0.1, color="grey", width=2, label=nothing)
    end

    plot!(x -> gcdyn.sigmoid(x, truth.λ_xscale, truth.λ_xshift, truth.λ_yscale, truth.λ_yshift); color="#1A4F87", width=5, label="Truth")
    title!("Birth rate (prior)")
    png("birth-rate-prior-samples.png")

    plot(xlims=(0, 10), ylims=(0, 6), dpi=300)

    for row in eachrow(posterior_samples[4500:5:end, :])
        plot!(x -> gcdyn.sigmoid(x, row.λ_xscale, row.λ_xshift, row.λ_yscale, row.λ_yshift); alpha=0.1, color="grey", width=2, label=nothing)
    end

    plot!(x -> gcdyn.sigmoid(x, truth.λ_xscale, truth.λ_xshift, truth.λ_yscale, truth.λ_yshift); color="#1A4F87", width=5, label="Truth")
    title!("Birth rate (posterior)")
    png("birth-rate-posterior-samples.png")

    plot(LogNormal(0, 0.5); xlims=(0, 3), label="Prior", fill=(0, 0.5), color="grey", width=0, dpi=300)
    histogram!(posterior_samples[:, :μ]; normalize=:pdf, label="Posterior", fill="lightblue", alpha=0.7)
    vline!([truth.μ]; label="Truth", color="#1A4F87", width=6)
    title!("Death rate")
    png("death-rate.png")

    plot(LogNormal(0, 0.5); xlims=(0, 4), label="Prior", fill=(0, 0.5), color="grey", width=0, dpi=300)
    histogram!(posterior_samples[:, :δ]; normalize=:pdf, label="Posterior", fill="lightblue", alpha=0.7)
    vline!([truth.δ]; label="Truth", color="#1A4F87", width=6)
    title!("Type change rate scalar")
    png("type-change-rate-scalar.png")

    print("Done!")    
end

main()
