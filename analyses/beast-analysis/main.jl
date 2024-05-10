println("Loading packages...")
using CSV, gcdyn, DataFrames, Distributions, JLD2, Optim, Turing

@model function SigmoidalModel(trees, Γ, type_space)
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
			present_time = maximum(node.time for node in LeafTraversal(tree))

			sampled_model = SigmoidalBranchingProcess(
				λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, ρ, 0, type_space, present_time
			)
			
			Turing.@addlogprob! loglikelihood(sampled_model, tree)
		end
	end
end

function main(i)
	# These are computed in a separate script
	type_space = [-0.4568874976254108, 0.0, 0.35395573620012577, 0.8642148598083795, 1.4959713153479637]
	discretization_table = Dict([1.1973728522068754, 3.1370222155629772] => 1.4959713153479637, [-0.0002917189005149, 0.09106099779661217] => 0.0, [0.5796933611135875, 1.1973728522068754] => 0.8642148598083795, [-5.7429821755606145, -0.0002917189005149] => -0.4568874976254108, [0.09106099779661217, 0.5796933611135875] => 0.35395573620012577)
	Γ = [-0.1413269484969319 0.014424334397623277 0.06274191355468922 0.043509467691191524 0.02065123285342786; 0.45826534761167814 -0.9318887771433151 0.1842969830395073 0.1644801031427861 0.12484634334934366; 0.3675105675562578 0.03143787246229119 -0.7945876303042252 0.20572502506025636 0.18991416522541987; 0.18531601882676527 0.02491133367835205 0.15843000625927553 -0.7575172015484248 0.388859842784032; 0.025285884854612837 0.006042798468735782 0.03807549714766527 0.0996768407609912 -0.16908102123200508]
			
	# Read in the trees and do inference
	println("Reading trees...")
	treeset = map(readdir("data/jld2-with-affinities/"; join=true)) do germinal_center_dir
		tree::TreeNode{Float64} = load_object(joinpath(germinal_center_dir, "tree-STATE_$i.jld2"))

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
	prior_samples = sample(SigmoidalModel(nothing, Γ, type_space), Prior(), 5000) |> DataFrame
	mkpath("out")
	CSV.write("out/samples-prior-$i.csv", prior_samples)

	println("Computing initial MCMC state...")
	model = SigmoidalModel(treeset, Γ, type_space)

	max_a_posteriori = optimize(model, MAP())

	open("out/map-$i.txt", "w") do f
		println(f, max_a_posteriori)
	end

	println("Sampling from posterior...")
	posterior_samples = sample(model, NUTS(), 1000, init_params=max_a_posteriori) |> DataFrame
	CSV.write("out/samples-posterior-$i.csv", posterior_samples)

	println("Done!")
end

main(50_000_000)
