println("Loading packages...")

using gcdyn, CSV, DataFrames, Turing, StatsPlots

println("Setting up model...")

priors = Dict(
    :xscale => Gamma(2, 1),
    :xshift => Normal(5, 1),
    :yscale => Gamma(2, 1),
    :yshift => Gamma(1, 1),
    :μ => LogNormal(0, 0.3),
    :γ => LogNormal(0, .5),
    :logit_p => Normal(0, 1.8),
    :logit_δ => Normal(1.4, 1.4),
    
    # The logit priors are used in MH, but these are used in visualization
    :p => Uniform(0, 1),
    :δ => Beta(3, 1)
)

# https://docs.julialang.org/en/v1/manual/performance-tips/index.html#Avoid-untyped-global-variables
const transition_p = 0.3
const transition_δ = 0.8
const truth = SigmoidalBirthRateBranchingProcess(1, 5, 1.5, 1, 1.3, 1.3, [2, 4, 6, 8], random_walk_transition_matrix([2, 4, 6, 8], transition_p; δ=transition_δ), 1, 0, 5)

@model function Model(trees::Vector{TreeNode})
    xscale  ~ priors[:xscale]
    xshift  ~ priors[:xshift]
    yscale  ~ priors[:yscale]
    yshift  ~ priors[:yshift]
    μ       ~ priors[:μ]
    γ       ~ priors[:γ]
    logit_p ~ priors[:logit_p]
    logit_δ ~ priors[:logit_δ]

    p = gcdyn.expit(logit_p)
    δ = gcdyn.expit(logit_p)

    sampled_model = SigmoidalBirthRateBranchingProcess(
        xscale, xshift, yscale, yshift, μ, γ, truth.state_space, random_walk_transition_matrix(truth.state_space, p; δ=δ), truth.ρ, truth.σ, truth.present_time
    )

    Turing.@addlogprob! loglikelihood(sampled_model, trees; reltol=1e-3, abstol=1e-3)
end

println("Sampling...")

# https://docs.julialang.org/en/v1/manual/style-guide/#Write-functions,-not-just-scripts
function run_simulations(num_treesets, num_trees, num_samples)
    chns = Vector{Chains}(undef, num_treesets)

    Threads.@threads for i in 1:num_treesets
        trees = rand_tree(truth, num_trees, truth.state_space[1]);

        chns[i] = sample(
            Model(trees),
            MH(
                :xscale => x -> LogNormal(log(x), 0.3),
                :xshift => x -> Normal(x, 1),
                :yscale => x -> LogNormal(log(x), 0.3),
                :yshift => x -> LogNormal(log(x), 0.3),
                :μ => x -> LogNormal(log(x), 0.3),
                :γ => x -> LogNormal(log(x), 0.3),
                :logit_p => x -> Normal(x, 0.5),
                :logit_δ => x -> Normal(x, 0.5)
            ),
            num_samples
        )
    end

    chns
end

chns = run_simulations(100, 68, 5000)

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
    :μ => median => :μ,
    :γ => median => :γ,
    :logit_p => (x -> gcdyn.expit(median(x))) => :p,
    :logit_δ => (x -> gcdyn.expit(median(x))) => :δ
)
select!(medians, Not(:run))

println("Visualizing...")

hists = map(propertynames(medians)) do param
    histogram(medians[!, param]; normalize=:pdf, label="Medians")

    true_value =
        if param == :p
            transition_p
        elseif param == :δ
            transition_δ
        else
            getfield(truth, param)
        end

    vline!([true_value]; label="Truth", linewidth=4)
    plot!(priors[param]; label="Prior", fill=(0, 0.5))
    title!(string(param))
end

plot(hists...;
    layout=(4, 2),
    thickness_scaling=0.75,
    dpi=300,
    size=(800, 600),
    plot_title="Posterior median sampling distribution"
)
png("posterior_medians.png")

println("Done!")
