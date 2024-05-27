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
	Γ = [-0.1406397792215961 0.025539064386011234 0.02239041261239341 0.03843979040291759 0.026195033505514945 0.016180571614424925 0.011894906700334; 0.4693851282614865 -0.9211499597373489 0.05971322799572612 0.1140424764180671 0.11208466566410887 0.08956984199358918 0.0763546194043711; 0.4013230896295922 0.067449258761276 -0.9176846372575828 0.13752154425215718 0.12290753818721405 0.09068178122349328 0.0978014252038502; 0.3115530411094381 0.0583519309668667 0.06015132971914894 -0.8639684580600837 0.14061015964262594 0.13598313427961448 0.15731886234238954; 0.19057925785983695 0.04564922014181959 0.05737333679026505 0.12996308029446996 -0.8583550082404436 0.1603958937223497 0.27439421943170245; 0.10054565371020249 0.03460641104444179 0.03600937365435159 0.10335157893002209 0.1367888544662057 -0.8373348510145002 0.4260329792092766; 0.01243071221073892 0.005567923177726808 0.007510221960654764 0.02421399149383519 0.0427953165171793 0.07549067936313324 -0.1680088447232682]
	
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