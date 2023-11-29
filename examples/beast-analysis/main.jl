println("Loading packages...")
using CSV, gcdyn, CategoricalArrays, DataFrames, Distributions, JLD2, JSON, StatsPlots, Turing

function main()
	println("Reading trees...")
    germinal_centers = Vector{Vector{TreeNode}}()

    for file in readdir("output/jld2/")
        trees = load_object(joinpath("output/jld2/", file))
        push!(germinal_centers, trees)
    end

	println("Binning states...")
	all_trees = (tree for gc in germinal_centers for tree in gc)
	bin_states!(all_trees)

	state_space = unique(node.state for tree in all_trees for node in PostOrderTraversal(tree)) |> sort
	present_time = maximum(leaf.t for tree in all_trees for leaf in LeafTraversal(tree))

	println("Sampling from prior...")
    prior_samples = sample(Model(missing, state_space, missing, present_time), Prior(), 100) |> DataFrame

	for i in 1:1 # 1:5
		println("[Treeset", i, "]")
		treeset = collect(sample(gc) for gc in germinal_centers)

		println("Sampling from posterior...")

		# BAD transition matrix
		transition_matrix = zeros(length(state_space), length(state_space))
		for tree in treeset[1:2]
			for node in PostOrderTraversal(tree)
				if node.event == :mutation
					j = findfirst(state_space .== node.state)
					i = findfirst(state_space .== node.up.state)
					transition_matrix[i, j] += 1
				end
			end
		end
		transition_matrix ./= sum(transition_matrix, dims=2)

		# Initial values

		posterior_samples = sample(
			Model(treeset, state_space, transition_matrix, present_time),
			Gibbs(
				MH(:xscale => x -> LogNormal(log(x), 0.2)),
				MH(:xshift => x -> Normal(x, 0.7)),
				MH(:yscale => x -> LogNormal(log(x), 0.2)),
				MH(:yshift => x -> LogNormal(log(x), 0.2)),
				MH(:μ => x -> LogNormal(log(x), 0.2)),
				MH(:γ => x -> LogNormal(log(x), 0.2))
			),
			1 # 5000
		) |> DataFrame

		println("Exporting samples...")
		CSV.write("posterior-samples-$i.csv", posterior_samples)

		println("Visualizing...")
		
		for (name, param) in zip(names(posterior_samples), eachcol(posterior_samples))
			plot(param; dpi=300)
			png("posterior-samples-$i-$name.png")
		end

		plot(xlims=(-10, 10), ylims=(-6, 6), dpi=300)
		for row in eachrow(prior_samples)
			plot!(x -> gcdyn.sigmoid(x, row.xscale, row.xshift, row.yscale, row.yshift); alpha=0.1, color="grey", width=2, label=nothing)
		end
		for row in eachrow(posterior_samples[4500:5:end, :])
			plot!(x -> gcdyn.sigmoid(x, row.xscale, row.xshift, row.yscale, row.yshift); alpha=0.1, color="blue", width=2, label=nothing)
		end
		title!("Birth rate")
		png("birth-rate-samples-treeset-$i.png")
	end

	println("Done!")
end

function bin_states!(trees; visualize=true)
	all_nodes = [node for tree in trees for node in PostOrderTraversal(tree)]

    # The bins are evenly spaced quantiles for the affinities in (-2, inf), including the min and max
	states = [node.state for node in all_nodes]
	num_bins = 5
	cutoffs = quantile(
		filter(x -> x >= -2, states),
		collect(0:(1/(num_bins-1)):1)
	)
	pushfirst!(cutoffs, minimum(states))

	binned_states = convert.(
        Float64,
        cut(states, cutoffs; extend=true, labels=(from, to, i; leftclosed, rightclosed) -> to)
    )
	
	for (node, binned_state) in zip(all_nodes, binned_states)
		node.state = binned_state
	end

    if visualize
        ENV["GKSwstype"] = "100" # operate plotting in headless mode
	    histogram(states; normalize=:pdf, fill="grey", alpha=0.2)
	    histogram!(binned_states; normalize=:pdf)
	    png("binned-states.png")
    end
end

@model function Model(trees, state_space, transition_matrix, present_time)
    xscale  ~ Gamma(2, 1)
    xshift  ~ Normal(0, 1)
    yscale  ~ Gamma(1, 1)
    yshift  ~ Gamma(1, 0.5)
    μ       ~ LogNormal(0, 0.1)
    γ       ~ LogNormal(0, 0.1)

    # Only compute loglikelihood if we're not sampling from the prior
    if DynamicPPL.getsampler(__context__) != DynamicPPL.SampleFromPrior()
		sampled_model = SigmoidalBirthRateBranchingProcess(
			xscale, xshift, yscale, yshift, μ, γ, state_space, transition_matrix, 1, 0, present_time
		)

        Turing.@addlogprob! loglikelihood(sampled_model, trees; reltol=1e-3, abstol=1e-3)
    end
end

main()

