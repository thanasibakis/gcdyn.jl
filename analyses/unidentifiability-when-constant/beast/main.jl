# If the truth is constant rate, the sigmoid is not identifiable.
# And seeing here, that might be the case.
# But this might not matter, because maybe all we care about is λ(x) and
# we can ignore the actual parameter values.
# The posterior histograms of parameters might look awful over different runs,
# but I think the birth rates themselves should roughly match

# Now, I also think that this identifiability is why the gradients are so big
# and unpredictable, which is making autodiff hard to do?
# Using Nelder-Mead here has been totally fine

using gcdyn, ADTypes, FillArrays, JLD2, LinearAlgebra, Optim, Random, Turing

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

@model function ConstantModel(trees, Γ, type_space, present_time)
	# Keep priors on the same scale for NUTS
	log_λ_base ~ Normal(0, 1)
	log_μ_base ~ Normal(0, 1)
	log_δ_base ~ Normal(0, 1)

	# Obtain our actual parameters from the proxies
	λ = exp(log_λ_base * 0.5)
	μ = exp(log_μ_base * 0.5)
	δ = exp(log_δ_base * 0.5)

	if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
		for tree in trees
			ρ = length(LeafTraversal(tree)) / 1000
			sampled_model = ConstantBranchingProcess(
				λ, μ, δ, Γ, ρ, 0, type_space, present_time
			)
			
			Turing.@addlogprob! loglikelihood(sampled_model, tree)
		end
	end
end

@model function DiscreteModel(trees, Γ, type_space, present_time)
	# Keep priors on the same scale for NUTS
	log_λ_base ~ MvNormal(Fill(0, length(type_space)), I)
	log_μ_base ~ Normal(0, 1)
	log_δ_base ~ Normal(0, 1)

	# Obtain our actual parameters from the proxies
	λ = @. exp(log_λ_base * 0.5)
	μ = exp(log_μ_base * 0.5)
	δ = exp(log_δ_base * 0.5)

	if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
		for tree in trees
			ρ = length(LeafTraversal(tree)) / 1000
			sampled_model = DiscreteBranchingProcess(
				λ, μ, δ, Γ, ρ, 0, type_space, present_time
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
	treeset = map(readdir("data/jld2-with-affinities/"; join=true)[1:2]) do germinal_center_dir
		tree::TreeNode{Float64} = load_object(joinpath(germinal_center_dir, "tree-STATE_50000000.jld2"))

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

	println("Type space: $(round.(type_space; digits=3))")
	println("Γ:")
	display(round.(Γ; digits=3))

	println()
	println("MLE of constant rate model")
	println("--------------------------")
	present_time = maximum(node.time for node in LeafTraversal(treeset[1]))
	model = ConstantModel([treeset[1]], Γ, type_space, present_time)

	for i in 1:10
		println("Run $i")
		o = optimize(model, MLE(), NelderMead(), Optim.Options(g_tol=1e-4))
		λ, μ, δ = @. round(exp(o.values * 0.5); digits=3)
		println("\tλ: $λ")
		println("\tμ: $μ")
		println("\tδ: $δ")
	end

	println()
	println("MAP of constant rate model")
	println("--------------------------")

	for i in 1:10
		println("Run $i")
		o = optimize(model, MAP(), NelderMead(), Optim.Options(g_tol=1e-4))
		λ, μ, δ = @. round(exp(o.values * 0.5); digits=3)
		println("\tλ: $λ")
		println("\tμ: $μ")
		println("\tδ: $δ")
	end

	println()
	println("MLE of discrete rate model")
	println("--------------------------")
	model = DiscreteModel([treeset[1]], Γ, type_space, present_time)

	for i in 1:10
		println("Run $i")
		o = optimize(model, MLE(), NelderMead(), Optim.Options(g_tol=1e-4))
		λ = round.(exp.(o.values[1:length(type_space)] .* 0.5); digits=3)
		μ, δ = @. round(exp(o.values[end-1:end] * 0.5); digits=3)
		println("\tλ: $λ")
		println("\tμ: $μ")
		println("\tδ: $δ")
	end

	println()
	println("MAP of discrete rate model")
	println("--------------------------")

	for i in 1:10
		println("Run $i")
		o = optimize(model, MAP(), NelderMead(), Optim.Options(g_tol=1e-4))
		λ = round.(exp.(o.values[1:length(type_space)] .* 0.5); digits=3)
		μ, δ = @. round(exp(o.values[end-1:end] * 0.5); digits=3)
		println("\tλ: $λ")
		println("\tμ: $μ")
		println("\tδ: $δ")
	end

	println()
	println("MLE of sigmoidal rate model")
	println("---------------------------")
	model = SigmoidalModel([treeset[1]], Γ, type_space, present_time)

	for i in 1:10
		println("Run $i")
		o = optimize(model, MLE(), NelderMead(), Optim.Options(g_tol=1e-4))
		log_λ_xscale_base, λ_xshift_base, log_λ_yscale_base, log_λ_yshift_base, log_μ_base, log_δ_base = o.values;
		λ_xscale = exp(log_λ_xscale_base * 0.75 + 0.5);
		λ_xshift = λ_xshift_base * sqrt(2);
		λ_yscale = exp(log_λ_yscale_base * 0.75 + 0.5);
		λ_yshift = exp(log_λ_yshift_base * 1.2 - 0.5);
		μ        = exp(log_μ_base * 0.5);
		δ        = exp(log_δ_base * 0.5);
		sampled_model = SigmoidalBranchingProcess(
			λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, length(LeafTraversal(treeset[1])) / 1000, 0, type_space, present_time
		);
		println("\tλ_xscale: $(round(λ_xscale; digits=3))")
		println("\tλ_xshift: $(round(λ_xshift; digits=3))")
		println("\tλ_yscale: $(round(λ_yscale; digits=3))")
		println("\tλ_yshift: $(round(λ_yshift; digits=3))")
		println("\tλ(type_space): $(round.(gcdyn.λ.(sampled_model, type_space); digits=3))")
		println("\tμ: $(round(μ; digits=3))")
		println("\tδ: $(round(δ; digits=3))")
	end

	println()
	println("MAP of sigmoidal rate model")
	println("---------------------------")

	for i in 1:10
		println("Run $i")
		o = optimize(model, MAP(), NelderMead(), Optim.Options(g_tol=1e-4))
		log_λ_xscale_base, λ_xshift_base, log_λ_yscale_base, log_λ_yshift_base, log_μ_base, log_δ_base = o.values;
		λ_xscale = exp(log_λ_xscale_base * 0.75 + 0.5);
		λ_xshift = λ_xshift_base * sqrt(2);
		λ_yscale = exp(log_λ_yscale_base * 0.75 + 0.5);
		λ_yshift = exp(log_λ_yshift_base * 1.2 - 0.5);
		μ        = exp(log_μ_base * 0.5);
		δ        = exp(log_δ_base * 0.5);
		sampled_model = SigmoidalBranchingProcess(
			λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, length(LeafTraversal(treeset[1])) / 1000, 0, type_space, present_time
		);
		println("\tλ_xscale: $(round(λ_xscale; digits=3))")
		println("\tλ_xshift: $(round(λ_xshift; digits=3))")
		println("\tλ_yscale: $(round(λ_yscale; digits=3))")
		println("\tλ_yshift: $(round(λ_yshift; digits=3))")
		println("\tλ(type_space): $(round.(gcdyn.λ.(sampled_model, type_space); digits=3))")
		println("\tμ: $(round(μ; digits=3))")
		println("\tδ: $(round(δ; digits=3))")
	end

	# Always specify chunk size, or else type instability
	# s = sample(model, NUTS(10, 0.65; adtype=AutoForwardDiff(chunksize=6)), 10; init_params=m.values)
end

main()
