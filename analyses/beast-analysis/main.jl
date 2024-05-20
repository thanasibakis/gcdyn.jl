println("Loading packages...")
using CSV, gcdyn, DataFrames, Distributions, JLD2, LinearAlgebra, Optim, Turing

@model function SigmoidalModel(trees, Γ, type_space)
	# Keep priors on the same scale for NUTS
	θ ~ MvNormal(zeros(6), I)

	# Obtain our actual parameters from the proxies
	λ_xscale := exp(θ[1] * 0.75 + 0.5)
	λ_xshift := θ[2] * sqrt(2)
	λ_yscale := exp(θ[3] * 0.75 + 0.5)
	λ_yshift := exp(θ[4] * 1.2 - 0.5)
	μ        := exp(θ[5] * 0.5)
	δ        := exp(θ[6] * 0.5)

	if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
		for tree in trees
			ρ = length(LeafTraversal(tree)) / 1000
			present_time = maximum(node.time for node in LeafTraversal(tree))

			sampled_model = SigmoidalBranchingProcess(
				λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, ρ, 0, type_space
			)
			
			Turing.@addlogprob! loglikelihood(sampled_model, tree, present_time)
		end
	end
end

@model function DiscreteModel(trees, Γ, type_space)
	# Keep priors on the same scale for NUTS
	θ  ~ MvNormal(zeros(length(type_space)), I)
	θ₂ ~ MvNormal(zeros(2), I)

	# Obtain our actual parameters from the proxies
	λ := @. exp(θ * 1.2 - 0.5)
	μ := exp(θ₂[1] * 0.5)
	δ := exp(θ₂[2] * 0.5)

	if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
		for tree in trees
			ρ = length(LeafTraversal(tree)) / 1000
			present_time = maximum(node.time for node in LeafTraversal(tree))

			sampled_model = DiscreteBranchingProcess(λ, μ, δ, Γ, ρ, 0, type_space)
			
			Turing.@addlogprob! loglikelihood(sampled_model, tree, present_time)
		end
	end
end

function main()
	out_path = "out/inference-with-5-types/"

	# These are computed in a separate script
	type_space = [-0.47337548031309196, 0.0, 0.3890161743465148, 0.8772905477925974, 1.5050315480924654]
	discretization_table = Dict([-0.0019916536956188, 0.1112097799378943] => 0.0, [1.2169193081838328, 3.1370222155629772] => 1.5050315480924654, [0.1112097799378943, 0.6232833274714342] => 0.3890161743465148, [-5.7429821755606145, -0.0019916536956188] => -0.47337548031309196, [0.6232833274714342, 1.2169193081838328] => 0.8772905477925974)
	Γ = [-0.13759621132325778 0.017438962978034704 0.062173694095601986 0.037631446426285416 0.020352107823335696; 0.4787917983478633 -0.9499610407021287 0.1915167193391453 0.16340854411275332 0.1162439789023668; 0.35831976008380606 0.04273374384233016 -0.7981733577914721 0.2049074077125957 0.19221244615274027; 0.17156133761426878 0.02961860828623697 0.1757694458953735 -0.7683034619386169 0.39135407014273765; 0.025444324868837295 0.00645327080006743 0.04523435532237741 0.09286563979906559 -0.16999759079034774]
		
	# Read in the trees and do inference
	println("Reading trees...")
	treeset = map(readdir("data/jld2-with-affinities/"; join=true)) do germinal_center_dir
		tree::TreeNode{Float64} = load_object(joinpath(germinal_center_dir, "tree-STATE_10000000.jld2"))

		map_types!(tree) do affinity
			for (bin, value) in discretization_table
				if bin[1] <= affinity < bin[2]
					return value
				end
			end
		
			if all(bin[2] <= affinity for bin in keys(discretization_table))
				return maximum(values(discretization_table))
			elseif all(affinity < bin[1] for bin in keys(discretization_table))
				return minimum(values(discretization_table))
			else
				error("Affinity $affinity not in any bin!")
			end
		end

		tree
	end

	println("Sampling from prior...")
	prior_samples = sample(DiscreteModel([], Γ, type_space), Prior(), 5000) |> DataFrame
	mkpath(out_path)
	CSV.write(joinpath(out_path, "samples-prior.csv"), prior_samples)

	println("Computing initial MCMC state...")
	model = DiscreteModel(treeset, Γ, type_space)

	max_a_posteriori = optimize(model, MAP())

	open(joinpath(out_path, "map.txt"), "w") do f
		println(f, max_a_posteriori)
	end

	println("Sampling from posterior...")
	posterior_samples = sample(model, NUTS(adtype=AutoForwardDiff(chunksize=2+length(type_space))), 1000, init_params=max_a_posteriori) |> DataFrame
	CSV.write(joinpath(out_path, "samples-posterior.csv"), posterior_samples)

	println("Done!")
end

main()
