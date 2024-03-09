println("Loading packages...")
using CSV, gcdyn, DataFrames, Distributions, JLD2, JSON, LinearAlgebra, StatsPlots, Turing

include("estimate_type_change_rate_matrix.jl")

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
	tc_rate_matrix = compute_rate_matrix(tenx_data.sequence, state_space)

	# Read in the trees and do inference
	println("Reading trees...")
    germinal_centers = Vector{Vector{TreeNode}}()

	#for file in readdir("data/jld2/")
    for file in readdir("data/jld2/")[1:2]
        trees = load_object(joinpath("data/jld2/", file))
        push!(germinal_centers, trees)
    end

	for gc in germinal_centers
		for tree in gc
			# This is my bad, I changed the event name after exporting JLD2 trees
			for node in PostOrderTraversal(tree)
				if node.event == :mutation
					node.event = :type_change
				end
			end

			map_states!(tree) do state
				get_discretization(state, discretization_table)
			end
		end
	end

	println("Sampling from posterior...")
	treeset = [germinal_centers[2][1]]
	present_time = maximum([node.t for tree in treeset for node in LeafTraversal(tree)])

	ρ = mean(length(LeafTraversal(tree)) / 1000 for tree in treeset)

	model = Model(treeset, tc_rate_matrix, ρ, state_space, present_time)
	posterior_samples = sample(
		model,
		Gibbs(
			MH(:λ => x -> LogNormal(log(x), 0.2)),
			MH(:μ => x -> LogNormal(log(x), 0.2)),
			MH(:δ => x -> LogNormal(log(x), 0.2))
		),
		10
	) |> DataFrame

	display(posterior_samples)

	println("Done!")
end

@model function Model(trees, Γ, ρ, state_space, present_time)
	λ		~ LogNormal(0, 1.2)
    μ       ~ LogNormal(0, 1.2)
    δ       ~ LogNormal(0, 1.2)

	if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
		sampled_model = VaryingTypeChangeRateBranchingProcess(0, 0, 0, λ, μ, δ, Γ, ρ, 0, state_space, present_time)
        
		Turing.@addlogprob! loglikelihood(sampled_model, trees; reltol=1e-3, abstol=1e-3)
    end
end
