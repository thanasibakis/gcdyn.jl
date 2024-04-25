println("Loading packages...")
using CSV, gcdyn, Distributions, JLD2, Optim, Turing

@model function SigmoidalModel(trees, Γ, type_space, present_time)
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
			ρ = length(LeafTraversal(tree)) / 1000
			sampled_model = SigmoidalBranchingProcess(
				λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, ρ, 0, type_space, present_time
			)
			
			Turing.@addlogprob! loglikelihood(sampled_model, tree)
		end
	end
end

function main()
	# These are computed in a separate script
	type_space = [-1.224171991580991, 0.0, 0.4104891647333908, 1.0158145231080074, 2.1957206408364116]
	discretization_table = Dict([0.7118957155228802, 1.414021615333377] => 1.0158145231080074, [1.414021615333377, 4.510046694952103] => 2.1957206408364116, [0.09499050830860184, 0.7118957155228802] => 0.4104891647333908, [-0.07325129835947444, 0.09499050830860184] => 0.0, [-8.4036117593617, -0.07325129835947444] => -1.224171991580991)
	Γ = [-0.118959461717717 0.021572790787430626 0.05847104833335269 0.029935094305333598 0.008980528291600079; 0.46692888970327073 -0.9168729322281818 0.21740604388300436 0.157495855653881 0.07504214298802565; 0.3339074698602106 0.06008653715185837 -0.7759426942221336 0.2187766224503561 0.1631720647597086; 0.12640020164010704 0.03902621761552961 0.18717669977384585 -0.7137820629490017 0.36117894391951927; 0.01258581593526034 0.005465553533638952 0.03941215667376345 0.09747739513205618 -0.15494092127471892]
	
	# Read in the trees and do inference
	println("Reading trees...")
	treeset = map(readdir("data/jld2/"; join=true)) do germinal_center_dir
		tree::TreeNode = load_object(joinpath(germinal_center_dir, "tree-5000.jld2"))

		map_types!(tree) do type
			get_discretization(type, discretization_table)
		end

		tree
	end

	# TODO: Why is this not already true
	for tree in treeset
		present_time = maximum(node.time for tree in treeset for node in LeafTraversal(tree))
		
		for leaf in LeafTraversal(tree)
			leaf.time = present_time
		end
	end

	println("Sampling from prior...")
	prior_samples = sample(SigmoidalModel(nothing, Γ, type_space, nothing), Prior(), 5000) |> DataFrame
	mkpath("out")
	CSV.write("out/samples-prior.csv", prior_samples)

	println("Computing initial MCMC state...")
	present_time = maximum(node.time for tree in treeset for node in LeafTraversal(tree))
	model = SigmoidalModel(treeset, Γ, type_space, present_time)

	max_a_posteriori = optimize(model, MAP())

	println("Sampling from posterior...")
	posterior_samples = sample(model, NUTS(), 1000, init_params=max_a_posteriori) |> DataFrame
	CSV.write("out/samples-posterior.csv", posterior_samples)

	println("Done!")
end

main()
