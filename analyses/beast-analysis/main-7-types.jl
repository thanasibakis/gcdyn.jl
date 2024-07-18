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
	out_path = "out/inference-with-7-types/"

	# These are computed in a separate script
	type_space = [-0.7659997821530665, -0.0489179662856651, 0.0, 0.3897703970482968, 0.760290064851131, 1.1507619185243034, 1.6496044812555084]
	discretization_table = Dict([0.5212510699994402, 0.9101370395508295] => 0.760290064851131, [0.16186154252713816, 0.5212510699994402] => 0.3897703970482968, [0.9101370395508295, 1.3508783131199409] => 1.1507619185243034, [1.3508783131199409, 3.1370222155629772] => 1.6496044812555084, [-5.7429821755606145, -0.16392105673653481] => -0.7659997821530665, [0.0, 0.16186154252713816] => 0.0, [-0.16392105673653481, 0.0] => -0.0489179662856651)
	Γ = [-0.13482607919002004 0.024285707008675536 0.021691261951706696 0.03500374232598943 0.027773321675420532 0.01586439354507176 0.010207652683156092; 0.4636159541877814 -0.923200465295669 0.06853453235819378 0.1194315012418524 0.11086468469707816 0.0826445831378219 0.07810920967294144; 0.4444165461633169 0.05869652496496638 -0.9870599300231081 0.13975363086896755 0.12577826778207082 0.1050148711958242 0.11340008904796225; 0.3158717597159614 0.06177851928978295 0.06488036963069256 -0.8692935580399166 0.15224915423297974 0.12717586397729377 0.1473378911932062; 0.19598018748998372 0.04347020267745612 0.058284070628991436 0.1374532665667048 -0.8558044370690243 0.16173829599545125 0.25887841371043696; 0.09556619976007796 0.032403281491132085 0.035220958142534876 0.10566287442760464 0.15168492640051687 -0.8288665482876542 0.4083283080657877; 0.011903034196159127 0.0057509041621892415 0.007021452756161283 0.02554471383670105 0.04119252283614619 0.07543046389476121 -0.1668430916821181]
	
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