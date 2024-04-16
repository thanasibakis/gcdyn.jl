using gcdyn, ADTypes, ForwardDiff, JLD2, Optim, Random, SciMLSensitivity, Turing, Zygote

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
			sampled_model = BranchingProcess(
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
			sampled_model = BranchingProcess(
				0, 0, 0, λ, μ, δ, Γ, ρ, 0, type_space, present_time
			)
			
			Turing.@addlogprob! loglikelihood(sampled_model, tree)
		end
	end
end

function main()
	treeset::Vector{TreeNode} = load_object("treeset.jld2")

	type_space::Vector{Float64} = load_object("type_space.jld2")
	Γ::Matrix{Float64}  = load_object("tc_rate_matrix.jld2")
	present_time = maximum(node.time for tree in treeset for node in LeafTraversal(tree))
	model = SigmoidalModel([treeset[1]], Γ, type_space, present_time)

	# TODO: Why is this not already true
	for tree in treeset
		for leaf in LeafTraversal(tree)
			leaf.time = present_time
		end
	end

	m = optimize(model, MLE(), NelderMead())

	# If you recreate sampled_model on this MLE, you'll basically see that λ(x) = 0.3018 for all x in type space...
	# and you'll get crazy local optima that still yield this constant value on the type space interval

	model2 = ConstantModel([treeset[1]], Γ, type_space, present_time)
	m2 = optimize(model, MLE(), NelderMead())

	# Yup.

	# I should try simulations where I simulate from a constant model and fit a sigmoid
	# to see if I get these issues / if I get crazy local optima there

	# Always specify chunk size, or else type instability
	# s = sample(model, NUTS(10, 0.65; adtype=AutoForwardDiff(chunksize=6)), 10; init_params=m.values)
end

main()
