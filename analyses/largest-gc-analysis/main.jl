println("Loading packages...")
using CSV, gcdyn, DataFrames, Distributions, JLD2, JSON, LinearAlgebra, Optim, Turing

include("estimate_type_change_rate_matrix.jl")

@model function Model(trees, Γ, type_space, present_time)
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
		for tree in trees
			ρ = length(LeafTraversal(tree)) / 1000 # + 0.1
			sampled_model = VaryingTypeChangeRateBranchingProcess(
				λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, ρ, 0, type_space, present_time
			)
			
			Turing.@addlogprob! loglikelihood(sampled_model, tree)
		end
    end
end

function main()
	# We will use the 10x sequences to determine the type bins and type change rate matrix.
	# Note that the heavy and light chain sequences each have an extra character and need correcting
	println("Computing type bins...")
	tenx_data = CSV.read("../../lib/gcreplay/analysis/output/10x/data.csv", DataFrame)
	tenx_data.sequence = chop.(tenx_data.nt_seq_H) .* chop.(tenx_data.nt_seq_L)
	tenx_data.time = tenx_data.var"time (days)"

	# Compute the type bins
	discretization_table = compute_discretization_table(tenx_data.delta_bind_CGG)
	type_space = values(discretization_table) |> collect |> sort

	# Compute the type change rate matrix
	println("Computing type change rate matrix...")
	tc_rate_matrix = compute_rate_matrix(tenx_data.sequence, discretization_table, type_space)

	# Read in the trees and do inference
	println("Reading trees...")
	treeset = map(readdir("data/jld2/"; join=true)) do germinal_center_dir
		tree::TreeNode = load_object(joinpath(germinal_center_dir, "tree-5000.jld2"))

		map_types!(tree) do type
			get_discretization(type, discretization_table)
		end

		tree
	end

	sort!(treeset; by=tree -> length(LeafTraversal(tree)), rev=true)
	treeset = treeset[1:23]

	println("Sampling from prior...")
	prior_samples = sample(Model(nothing, tc_rate_matrix, type_space, nothing), Prior(), 5000) |> DataFrame
	mkpath("out")
	CSV.write("out/samples-prior.csv", prior_samples)

	Threads.@threads for (i, tree) in collect(enumerate(treeset))
		present_time = maximum(node.time for tree in treeset for node in LeafTraversal(tree))
		model = Model([tree], tc_rate_matrix, type_space, present_time)

		posterior_samples = sample(
			model,
			Gibbs(
				MH(:log_λ_xscale_base => x -> Normal(x, 0.1)),
				MH(:λ_xshift_base     => x -> Normal(x, 0.1)),
				MH(:log_λ_yscale_base => x -> Normal(x, 0.1)),
				MH(:log_λ_yshift_base => x -> Normal(x, 0.1)),
				MH(:log_μ_base        => x -> Normal(x, 0.1)),
				MH(:log_δ_base        => x -> Normal(x, 0.1)),
			),
				10000,
		) |> DataFrame
		CSV.write("out/samples-posterior-$i.csv", posterior_samples)
	end

	println("Done!")
end

main()
