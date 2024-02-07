println("Loading packages...")

using gcdyn, CSV, DataFrames, Turing, StatsPlots

@model function Model(trees, Γ, state_space, present_time)
    λ_xscale  ~ Gamma(2, 1)
    λ_xshift  ~ Normal(5, 1)
    λ_yscale  ~ Gamma(2, 1)
    λ_yshift  ~ Gamma(1, 1)
    μ       ~ LogNormal(0, 0.5)
    δ       ~ LogNormal(0, 0.5)

    sampled_model = VaryingTypeChangeRateBranchingProcess(
        λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, 1, 0, state_space, present_time
    )

    # TODO: remove the ismissing check once you learn how Turing prior sampling works
    if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext() && !ismissing(trees)
        Turing.@addlogprob! loglikelihood(sampled_model, trees)
    end
end

function main()
	println("Setting up model...")

    Γ = [-4 2 1 1; 1 -4 2 1; 2 1 -4 1; 1 2 1 -4]
    truth = VaryingTypeChangeRateBranchingProcess(1, 5, 1.5, 1, 1.3, 0.25, Γ, 1, 0, [2, 4, 6, 8], 3)

	println("Sampling from prior...")
    prior_samples = sample(Model(missing, truth.Γ, truth.state_space, truth.present_time), Prior(), 100) |> DataFrame

	println("Sampling from posterior...")
    num_treesets = 100
	num_trees_per_set = 15
    dfs = Vector{DataFrame}(undef, num_treesets)

    Threads.@threads for i in 1:num_treesets
        treeset = rand_tree(truth, num_trees_per_set, truth.state_space[1]);

        dfs[i] = sample(
            Model(treeset, truth.Γ, truth.state_space, truth.present_time),
            MH(
                :λ_xscale => x -> LogNormal(log(x), 0.2),
                :λ_xshift => x -> Normal(x, 0.7),
                :λ_yscale => x -> LogNormal(log(x), 0.2),
                :λ_yshift => x -> LogNormal(log(x), 0.2),
                :μ => x -> LogNormal(log(x), 0.2),
                :δ => x -> LogNormal(log(x), 0.2),
            ),
            # Gibbs(
            #     MH(:λ_xscale => x -> LogNormal(log(x), 0.2)),
            #     MH(:λ_xshift => x -> Normal(x, 0.7)),
            #     MH(:λ_yscale => x -> LogNormal(log(x), 0.2)),
            #     MH(:λ_yshift => x -> LogNormal(log(x), 0.2)),
            #     MH(:μ => x -> LogNormal(log(x), 0.2)),
            #     MH(:δ => x -> LogNormal(log(x), 0.2)),
            # ),
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
        :λ_xscale => median => :λ_xscale,
        :λ_xshift => median => :λ_xshift,
        :λ_yscale => median => :λ_yscale,
        :λ_yshift => median => :λ_yshift,
        :μ => median => :μ,
        :δ => median => :δ
    )
    select!(medians, Not(:run))

    quantiles_025 = combine(
        groupby(posterior_samples, :run),
        :λ_xscale => (x -> quantile(x, 0.025)) => :λ_xscale,
        :λ_xshift => (x -> quantile(x, 0.025)) => :λ_xshift,
        :λ_yscale => (x -> quantile(x, 0.025)) => :λ_yscale,
        :λ_yshift => (x -> quantile(x, 0.025)) => :λ_yshift,
        :μ => (x -> quantile(x, 0.025)) => :μ,
        :δ=> (x -> quantile(x, 0.025)) => :δ,
    )
    select!(quantiles_025, Not(:run))

    quantiles_975 = combine(
        groupby(posterior_samples, :run),
        :λ_xscale => (x -> quantile(x, 0.975)) => :λ_xscale,
        :λ_xshift => (x -> quantile(x, 0.975)) => :λ_xshift,
        :λ_yscale => (x -> quantile(x, 0.975)) => :λ_yscale,
        :λ_yshift => (x -> quantile(x, 0.975)) => :λ_yshift,
        :μ => (x -> quantile(x, 0.975)) => :μ,
        :δ => (x -> quantile(x, 0.975)) => :δ,
    )
    select!(quantiles_975, Not(:run))

    median_hists = []
    error_hists = []
    ci_length_hists = []

    for param in propertynames(medians)
        true_value = getfield(truth, param)

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
            true_value = getfield(truth, param)
            coverage = quantiles_025[:, param] .<= true_value .<= quantiles_975[:, param]
            println(file, string(param), ": ", mean(coverage))
        end
    end

	println("Done!")
end

main()
