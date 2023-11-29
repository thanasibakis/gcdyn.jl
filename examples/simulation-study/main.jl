println("Loading packages...")

using gcdyn, CSV, DataFrames, Turing, StatsPlots

function main()
	println("Setting up model...")
    transition_p = 0.3
    transition_δ = 0.8
    truth = SigmoidalBirthRateBranchingProcess(1, 5, 1.5, 1, 1.3, 1.3, [2, 4, 6, 8], random_walk_transition_matrix([2, 4, 6, 8], transition_p; δ=transition_δ), 1, 0, 3)

	println("Sampling from prior...")
    prior_samples = sample(Model(Vector{TreeNode}(undef, 0), truth.state_space, truth.present_time), Prior(), 100) |> DataFrame
    prior_samples.p = gcdyn.expit.(prior_samples[:, :logit_p])
    prior_samples.δ = gcdyn.expit.(prior_samples[:, :logit_δ])

	println("Sampling from posterior...")
    num_treesets = 100
	num_trees_per_set = 15
    dfs = Vector{DataFrame}(undef, num_treesets)

    Threads.@threads for i in 1:num_treesets
        treeset = rand_tree(truth, num_trees_per_set, truth.state_space[1]);

        dfs[i] = sample(
            Model(treeset, truth.state_space, truth.present_time),
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

    quantiles_025 = combine(
        groupby(posterior_samples, :run),
        :xscale => (x -> quantile(x, 0.025)) => :xscale,
        :xshift => (x -> quantile(x, 0.025)) => :xshift,
        :yscale => (x -> quantile(x, 0.025)) => :yscale,
        :yshift => (x -> quantile(x, 0.025)) => :yshift,
        :μ => (x -> quantile(x, 0.025)) => :μ,
        :γ => (x -> quantile(x, 0.025)) => :γ,
        :logit_p => (x -> gcdyn.expit(quantile(x, 0.025))) => :p,
        :logit_δ => (x -> gcdyn.expit(quantile(x, 0.025))) => :δ
    )
    select!(quantiles_025, Not(:run))

    quantiles_975 = combine(
        groupby(posterior_samples, :run),
        :xscale => (x -> quantile(x, 0.975)) => :xscale,
        :xshift => (x -> quantile(x, 0.975)) => :xshift,
        :yscale => (x -> quantile(x, 0.975)) => :yscale,
        :yshift => (x -> quantile(x, 0.975)) => :yshift,
        :μ => (x -> quantile(x, 0.975)) => :μ,
        :γ => (x -> quantile(x, 0.975)) => :γ,
        :logit_p => (x -> gcdyn.expit(quantile(x, 0.975))) => :p,
        :logit_δ => (x -> gcdyn.expit(quantile(x, 0.975))) => :δ
    )
    select!(quantiles_975, Not(:run))

    median_hists = []
    error_hists = []
    ci_length_hists = []

    for param in propertynames(medians)
        true_value =
            if param == :p
                transition_p
            elseif param == :δ
                transition_δ
            else
                getfield(truth, param)
            end

        median_hist = histogram(prior_samples[:, param]; normalize=:pdf, label="Prior", fill="grey")
        histogram!(medians[!, param]; normalize=:pdf, fill="lightblue", alpha=0.7, label="Medians")
        vline!([true_value]; label="Truth", color="#1A4F87", width=6)
        title!(string(param))
        push!(median_hists, median_hist)

        relative_errors = (medians[:, param] .- true_value) ./ true_value
        error_hist = histogram(relative_errors; normalize=:pdf, fill="lightblue", alpha=0.7, label="Errors")
        xlims!((-3, 3))
        title!(string(param))
        push!(error_hists, error_hist)

        ci_lengths = (quantiles_975[:, param] .- quantiles_025[:, param]) ./ true_value
        ci_length_hist = histogram(ci_lengths; normalize=:pdf, fill="lightblue", alpha=0.7, label="CI length")
        xlims!((0, xlims()[2]))
        title!(string(param))
        push!(ci_length_hists, ci_length_hist)
    end
    
    plot(median_hists...;
        layout=(2, 4),
        thickness_scaling=2.25,
        size=(3600, 1200),
        plot_title="Posterior median sampling distribution"
    )

    png("posterior-medians.png")
    
    plot(error_hists...;
        layout=(2, 4),
        thickness_scaling=2.25,
        size=(3600, 1200),
        plot_title="Relative error distribution of posterior median"
    )

    png("relative-errors.png")
    
    plot(ci_length_hists...;
        layout=(2, 4),
        thickness_scaling=2.25,
        size=(3600, 1200),
        plot_title="Length distribution of 95% CIs (normalized)"
    )

    png("ci-lengths.png")

    open("ci-coverage-proportions.txt", "w") do file
        println(file, "95% Coverage proportions", "\n")
        map(propertynames(quantiles_025)) do param
            true_value =
                if param == :p
                    transition_p
                elseif param == :δ
                    transition_δ
                else
                    getfield(truth, param)
                end

            coverage = quantiles_025[:, param] .<= true_value .<= quantiles_975[:, param]
            println(file, string(param), ": ", mean(coverage))
        end
    end

	println("Done!")
end

@model function Model(trees::Vector{TreeNode}, state_space, present_time)
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
        xscale, xshift, yscale, yshift, μ, γ, state_space, random_walk_transition_matrix(state_space, p; δ=δ), 1, 0, present_time
    )

    # Only compute loglikelihood if we're not sampling from the prior
    if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
        Turing.@addlogprob! loglikelihood(sampled_model, trees; reltol=1e-3, abstol=1e-3)
    end
end

main()
