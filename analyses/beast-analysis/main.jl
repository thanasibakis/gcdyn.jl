println("Loading packages...")
using CSV, gcdyn, DataFrames, Distributions, JLD2, JSON, LinearAlgebra, Turing

include("estimate_type_change_rate_matrix.jl")

@model function Model(trees, Γ, ρ, state_space, present_time)
    # Keep priors on the same scale for NUTS
    log_λ_xscale_base ~ Normal(0, 1)
    λ_xshift_base     ~ Normal(0, 1)
    log_λ_yscale_base ~ Normal(0, 1)
    log_λ_yshift_base ~ Normal(0, 1)
    log_μ_base        ~ Normal(0, 1)
    log_δ_base        ~ Normal(0, 1)

    # Obtain our actual parameters from the proxies
    λ_xscale = exp(log_λ_xscale_base * 0.75 + 0.5)
    λ_xshift = λ_xshift_base * sqrt(2)
    λ_yscale = exp(log_λ_yscale_base * 0.75 + 0.5)
    λ_yshift = exp(log_λ_yshift_base * 1.2 - 0.5)
    μ        = exp(log_μ_base * 0.5)
    δ        = exp(log_δ_base * 0.5)

    if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
		sampled_model = VaryingTypeChangeRateBranchingProcess(
			λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, ρ, 0, state_space, present_time
		)
		
        Turing.@addlogprob! loglikelihood(sampled_model, trees)
    end
end

function main()
	# We will use the 10x sequences to determine the state bins and type change rate matrix.
	# Note that the heavy and light chain sequences each have an extra character and need correcting
	println("Computing state bins...")
	tenx_data = CSV.read("../../lib/gcreplay/analysis/output/10x/data.csv", DataFrame)
	tenx_data.sequence = chop.(tenx_data.nt_seq_H) .* chop.(tenx_data.nt_seq_L)
	tenx_data.time = tenx_data.var"time (days)"

	# Compute the state bins
	discretization_table = compute_discretization_table(tenx_data.delta_bind_CGG)
	state_space = values(discretization_table) |> collect |> sort

	# Compute the type change rate matrix
	println("Computing type change rate matrix...")
	tc_rate_matrix = compute_rate_matrix(tenx_data.sequence, discretization_table, state_space)

	# Read in the trees and do inference
	println("Reading trees...")
	germinal_centers = map(readdir("data/jld2/"; join=true)[1:5]) do file
		trees::Vector{TreeNode} = load_object(file)

		for tree in trees
			# TODO: This is my bad, I changed the event name after exporting JLD2 trees
			for node in PostOrderTraversal(tree)
				if node.event == :mutation
					node.event = :type_change
				end
			end

			# This part is fine, don't remove in the above TODO
			map_states!(tree) do state
				get_discretization(state, discretization_table)
			end
		end

		trees
	end

	println("Sampling from prior...")
	prior_samples = sample(Model(nothing, tc_rate_matrix, nothing, state_space, nothing), Prior(), 100) |> DataFrame
	CSV.write("samples-prior.csv", prior_samples)

	Threads.@threads for i in eachindex(germinal_centers)
		name = "posterior-$i"
		println("Sampling from $name...")

		treeset = [germinal_centers[i][end]]
		present_time = maximum([node.t for tree in treeset for node in LeafTraversal(tree)])
		ρ = mean(length(LeafTraversal(tree)) / 1000 for tree in treeset)

		model = Model(treeset, tc_rate_matrix, ρ, state_space, present_time)
		posterior_samples = sample(model, NUTS(), 5000) |> DataFrame
		CSV.write("samples-$name.csv", posterior_samples)
	end

	println("Done!")
end

main()