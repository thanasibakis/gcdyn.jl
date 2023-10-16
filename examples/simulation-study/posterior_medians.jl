println("Loading packages...")

using gcdyn, CSV, DataFrames, Turing, StatsPlots

@model function Model(trees::Vector{TreeNode}, truth::AbstractBranchingProcess)
    xscale  ~ Gamma(2, 1)
    xshift  ~ Normal(5, 1)
    yscale  ~ Gamma(2, 1)
    yshift  ~ Gamma(1, 1)
    μ       ~ LogNormal(0, 0.5)
    γ       ~ LogNormal(0, 0.5)
    logit_p ~ Normal(0, 1.8)
    logit_δ ~ Normal(1.4, 1.4)

    p = gcdyn.expit(logit_p)  # roughly Uniform(0, 1)
    δ = gcdyn.expit(logit_p)  # roughly Beta(3, 1)

    sampled_model = SigmoidalBirthRateBranchingProcess(
        xscale, xshift, yscale, yshift, μ, γ, truth.state_space, random_walk_transition_matrix(truth.state_space, p; δ=δ), truth.ρ, truth.σ, truth.present_time
    )

    Turing.@addlogprob! loglikelihood(sampled_model, trees; reltol=1e-3, abstol=1e-3)
end

function main()
    println("Setting up model...")

    transition_p = 0.3
    transition_δ = 0.8
    truth = SigmoidalBirthRateBranchingProcess(1, 5, 1.5, 1, 1.3, 1.3, [2, 4, 6, 8], random_walk_transition_matrix([2, 4, 6, 8], transition_p; δ=transition_δ), 1, 0, 3)

    println("Sampling from prior...")

    prior_samples = sample(Model(rand_tree(truth, 1, truth.state_space[1]), truth), Prior(), 100) |> DataFrame
    prior_samples.p = gcdyn.expit.(prior_samples[:, :logit_p])
    prior_samples.δ = gcdyn.expit.(prior_samples[:, :logit_δ])

    println("Sampling from posterior...")

    num_treesets = 100
    dfs = Vector{DataFrame}(undef, num_treesets)

    Threads.@threads for i in 1:num_treesets
        trees = rand_tree(truth, 15, truth.state_space[1]);

        dfs[i] = sample(
            Model(trees, truth),
            MH(
                :xscale => x -> LogNormal(log(x), 0.2),
                :xshift => x -> Normal(x, 0.7),
                :yscale => x -> LogNormal(log(x), 0.2),
                :yshift => x -> LogNormal(log(x), 0.2),
                :μ => x -> LogNormal(log(x), 0.2),
                :γ => x -> LogNormal(log(x), 0.2),
                :logit_p => x -> Normal(x, 0.2),
                :logit_δ => x -> Normal(x, 0.2)
            ),
            5000
        ) |> DataFrame

        dfs[i].run .= i
    end

    println("Exporting samples...")
    
    posterior_samples = vcat(dfs...)
    CSV.write("posterior-samples.csv", posterior_samples)

    println("Visualizing...")

    medians = combine(
        groupby(posterior_samples, :run),
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

    hists = map(propertynames(medians)) do param
        histogram(prior_samples[:, param]; normalize=:pdf, label="Prior", fill="grey")
        histogram!(medians[!, param]; normalize=:pdf, fill="lightblue", alpha=0.7, label="Medians")
    
        true_value =
            if param == :p
                transition_p
            elseif param == :δ
                transition_δ
            else
                getfield(truth, param)
            end
    
        vline!([true_value]; label="Truth", color="#1A4F87", width=6)
        title!(string(param))
    end
    
    plot(hists...;
        layout=(2, 4),
        thickness_scaling=2.25,
        size=(3600, 1200),
        plot_title="Posterior median sampling distribution"
    )

    png("posterior-medians.png")
    
    println("Done!")
end

main()
