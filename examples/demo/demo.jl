println("Loading packages...")

using gcdyn, CSV, DataFrames, Turing, StatsPlots

@model function Model(trees::Vector{TreeNode}, truth::AbstractBranchingProcess)
    xscale  ~ Gamma(2, 1)
    xshift  ~ Normal(5, 1)
    yscale  ~ Gamma(2, 1)
    yshift  ~ Gamma(1, 1)
    μ       ~ LogNormal(0, 0.5)
    γ       ~ LogNormal(0, 0.5)

    sampled_model = SigmoidalBirthRateBranchingProcess(
        xscale, xshift, yscale, yshift, μ, γ, truth.state_space, uniform_transition_matrix(truth.state_space), truth.ρ, truth.σ, truth.present_time
    )

    Turing.@addlogprob! loglikelihood(sampled_model, trees; reltol=1e-3, abstol=1e-3)
end

function main()
    println("Setting up model...")

    truth = SigmoidalBirthRateBranchingProcess(1, 5, 1.5, 1, 1.3, 1.3, [2, 4, 6, 8], uniform_transition_matrix([2, 4, 6, 8]), 1, 0, 3)
    trees = rand_tree(truth, 15, truth.state_space[1]);
    model = Model(trees, truth)

    num_leaves = sort(length(LeafTraversal(tree)) for tree in trees)
    num_nodes = sort(length(PostOrderTraversal(tree)) for tree in trees)

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
        MH(
            :xscale => x -> LogNormal(log(x), 0.2),
            :xshift => x -> Normal(x, 0.7),
            :yscale => x -> LogNormal(log(x), 0.2),
            :yshift => x -> LogNormal(log(x), 0.2),
            :μ => x -> LogNormal(log(x), 0.2),
            :γ => x -> LogNormal(log(x), 0.2),
        ),
        5000
    ) |> DataFrame

    println("Exporting samples...")

    CSV.write("posterior-samples.csv", posterior_samples)

    println("Visualizing...")

    plot(xlims=(0, 10), ylims=(0, 6), dpi=300)

    for row in eachrow(prior_samples)
        plot!(x -> gcdyn.sigmoid(x, row.xscale, row.xshift, row.yscale, row.yshift); alpha=0.1, color="grey", width=2, label=nothing)
    end

    plot!(x -> gcdyn.sigmoid(x, truth.xscale, truth.xshift, truth.yscale, truth.yshift); color="#1A4F87", width=5, label="Truth")
    title!("Birth rate (prior)")

    png("birth-rate-prior-samples.png")

    plot(xlims=(0, 10), ylims=(0, 6), dpi=300)

    for row in eachrow(posterior_samples[4500:5:end, :])
        plot!(x -> gcdyn.sigmoid(x, row.xscale, row.xshift, row.yscale, row.yshift); alpha=0.1, color="grey", width=2, label=nothing)
    end

    plot!(x -> gcdyn.sigmoid(x, truth.xscale, truth.xshift, truth.yscale, truth.yshift); color="#1A4F87", width=5, label="Truth")
    title!("Birth rate (posterior)")

    png("birth-rate-posterior-samples.png")

    plot(LogNormal(0, 0.5); xlims=(0, 3), label="Prior", fill=(0, 0.5), color="grey", width=0, dpi=300)
    histogram!(posterior_samples[:, :μ]; normalize=:pdf, label="Posterior", fill="lightblue", alpha=0.7)
    vline!([truth.μ]; label="Truth", color="#1A4F87", width=6)
    title!("Death rate")

    png("death-rate.png")

    plot(LogNormal(0, 0.5); xlims=(0, 4), label="Prior", fill=(0, 0.5), color="grey", width=0, dpi=300)
    histogram!(posterior_samples[:, :γ]; normalize=:pdf, label="Posterior", fill="lightblue", alpha=0.7)
    vline!([truth.γ]; label="Truth", color="#1A4F87", width=6)
    title!("Mutation rate")

    png("mutation-rate.png")

    print("Done!")    
end

main()
