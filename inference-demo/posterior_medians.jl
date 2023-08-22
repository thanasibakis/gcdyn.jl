println("Loading packages...")

using gcdyn, CSV, DataFrames, Turing, StatsPlots

println("Setting up model...")

# https://docs.julialang.org/en/v1/manual/performance-tips/index.html#Avoid-untyped-global-variables
const truth = SigmoidalBirthRateBranchingProcess(1, 5, 1.5, 1, 1.3, 2, [2, 4, 6, 8], 1, 0, 2)

@model function FullModel(trees::Vector{TreeNode})
    xscale ~ Gamma(2, 1)
    xshift ~ Normal(5, 1)
    yscale ~ Gamma(2, 1)
    yshift ~ Gamma(1, 1)
    # λ ~ LogNormal(1.5, 1)

    μ ~ LogNormal(0, 0.3)
    # γ ~ LogNormal(1.5, 1)

    sampled_model = SigmoidalBirthRateBranchingProcess(
        xscale, xshift, yscale, yshift, μ, truth.γ, truth.state_space, truth.transition_matrix, truth.ρ, truth.σ, truth.present_time
    )

    Turing.@addlogprob! loglikelihood(sampled_model, trees; reltol=1e-3, abstol=1e-3)
end

@model function AppxModel(trees::Vector{TreeNode})
    xscale ~ Gamma(2, 1)
    xshift ~ Normal(5, 1)
    yscale ~ Gamma(2, 1)
    yshift ~ Gamma(1, 1)
    # λ ~ LogNormal(1.5, 1)

    μ ~ LogNormal(0, 0.3)
    # γ ~ LogNormal(1.5, 1)

    sampled_model = SigmoidalBirthRateBranchingProcess(
        xscale, xshift, yscale, yshift, μ, truth.γ, truth.state_space, truth.transition_matrix, truth.ρ, truth.σ, truth.present_time
    )

    Turing.@addlogprob! sum(gcdyn.stadler_appx_loglikelhood(sampled_model, tree) for tree in trees)
end

println("Sampling...")

# https://docs.julialang.org/en/v1/manual/style-guide/#Write-functions,-not-just-scripts
function run_simulations(num_treesets, num_trees, num_samples)
    chns = Vector{Chains}(undef, num_treesets)

    Threads.@threads for i in 1:num_treesets
        trees = rand_tree(truth, num_trees, truth.state_space[1]);

        chns[i] = sample(
            FullModel(trees),
            MH(
                :xscale => x -> LogNormal(log(x), 0.3),
                :xshift => x -> Normal(x, 1),
                :yscale => x -> LogNormal(log(x), 0.3),
                :yshift => x -> LogNormal(log(x), 0.3),
                :μ => x -> LogNormal(log(x), 0.3)
            ),
            num_samples
        )
    end

    chns
end

chns = run_simulations(100, 20, 1000)

println("Exporting samples...")

dfs = map(enumerate(chns)) do (i, chn)
    df = DataFrame(chn)
    df.run .= i
    df
end

df = vcat(dfs...)
CSV.write("samples.csv", df)

medians = combine(
    groupby(df, :run),
    :xscale => median => :xscale,
    :xshift => median => :xshift,
    :yscale => median => :yscale,
    :yshift => median => :yshift,
    :μ => median => :μ
)

println("Visualizing...")

hists = map((:xscale, :xshift, :yscale, :yshift, :μ)) do param
    histogram(medians[!, param]; normalize=:pdf, label="Medians")
    vline!([getfield(truth, param)]; label="Truth", linewidth=4)
    title!(string(param))
end

plot(hists...; layout=(3, 2), thickness_scaling=0.75, dpi=300, size=(600, 600), plot_title="Posterior median sampling distribution")
png("posterior_medians.png")

println("Done!")
