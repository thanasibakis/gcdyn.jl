println("Loading packages...")

using gcdyn, CSV, DataFrames, Turing, StatsPlots

println("Setting up model...")

const true_params = Dict(
    :xscale => 1,
    :xshift => 5,
    :yscale => 1.5,
    :yshift => 1,
    :μ => 1.3
)

expit(x) = 1 / (1 + exp(-x))
sigmoid(x, xscale, xshift, yscale, yshift) = yscale * expit(xscale * (x - xshift)) + yshift

const λ_truth = x -> sigmoid(x, true_params[:xscale], true_params[:xshift], true_params[:yscale], true_params[:yshift])
const truth = MultitypeBranchingProcess(λ_truth, true_params[:mu], 2, [2, 4, 6, 8], 1, 0, 2)

@model function FullModel(trees::Vector{TreeNode})
    xscale ~ Gamma(2, 1)
    xshift ~ Normal(5, 1)
    yscale ~ Gamma(2, 1)
    yshift ~ Gamma(1, 1)

    μ ~ LogNormal(0, 0.3)
    # γ ~ LogNormal(1.5, 1)

    λ = x -> sigmoid(x, xscale, xshift, yscale, yshift)

    sampled_model = MultitypeBranchingProcess(
        λ, μ, truth.γ, truth.state_space, truth.transition_matrix, truth.ρ, truth.σ, truth.present_time
    )

    Turing.@addlogprob! loglikelihood(sampled_model, trees)
end

@model function AppxModel(trees::Vector{TreeNode})
    xscale ~ Gamma(2, 1)
    xshift ~ Normal(5, 1)
    yscale ~ Gamma(2, 1)
    yshift ~ Gamma(1, 1)

    μ ~ LogNormal(0, 0.3)
    # γ ~ LogNormal(1.5, 1)

    λ = x -> sigmoid(x, xscale, xshift, yscale, yshift)

    sampled_model = MultitypeBranchingProcess(
        λ, μ, truth.γ, truth.state_space, truth.transition_matrix, truth.ρ, truth.σ, truth.present_time
    )

    Turing.@addlogprob! sum(stadler_appx_loglikelhood(sampled_model, tree) for tree in trees)
end

println("Sampling...")

NUM_TREESETS = 100

chns = Vector{Chains}(undef, NUM_TREESETS)

Threads.@threads for i in 1:NUM_TREESETS
    trees = rand_tree(truth, 30, truth.state_space[1]);

    chns[i] = sample(
        AppxModel(trees),
        MH(
            :xscale => x -> LogNormal(log(x), 1),
            :xshift => x -> Normal(x, 1),
            :yscale => x -> LogNormal(log(x), 1),
            :yshift => x -> LogNormal(log(x), 1),
            :μ => x -> LogNormal(log(x), 0.3)
        ),
        1000
    )
end

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
    vline!([true_params[param]]; label="Truth", linewidth=4)
    title!(string(param))
end

plot(hists...; layout=(3, 2), thickness_scaling=0.75, dpi=300, size=(600, 600), plot_title="Posterior median sampling distribution")
png("posterior_medians.png")

println("Done!")
