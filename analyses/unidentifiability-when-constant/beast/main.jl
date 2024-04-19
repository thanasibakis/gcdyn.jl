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

function main(treeset_path, type_space_path, Γ_path)
	treeset::Vector{TreeNode} = load_object(treeset_path)

	type_space::Vector{Float64} = load_object(type_space_path)
	Γ::Matrix{Float64}  = load_object(Γ_path)
	present_time = maximum(node.time for tree in treeset for node in LeafTraversal(tree))
	
	# TODO: Why is this not already true
	for tree in treeset
		for leaf in LeafTraversal(tree)
			leaf.time = present_time
		end
	end

	println("Type space: $(round.(type_space; digits=3))")
	println("Γ:")
	display(round.(Γ; digits=3))

	println()
	println("MLE of constant rate model")
	println("--------------------------")
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

main("treeset-small.jld2", "type_space-small.jld2", "tc_rate_matrix-small.jld2")
println("\n-----------------------------------\n")
main("treeset.jld2", "type_space.jld2", "tc_rate_matrix.jld2")