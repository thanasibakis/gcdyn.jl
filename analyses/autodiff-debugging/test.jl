using gcdyn, JLD2, Optim, Random, SciMLSensitivity, Turing, Zygote

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

function objective(params, tree, Γ, type_space)
	log_λ_xscale_base = params[1]
    λ_xshift_base     = params[2]
    log_λ_yscale_base = params[3]
    log_λ_yshift_base = params[4]
    log_μ_base        = params[5]
    log_δ_base        = params[6]

	λ_xscale = exp(log_λ_xscale_base * 0.75 + 0.5)
    λ_xshift = λ_xshift_base * sqrt(2)
    λ_yscale = exp(log_λ_yscale_base * 0.75 + 0.5)
    λ_yshift = exp(log_λ_yshift_base * 1.2 - 0.5)
    μ        = exp(log_μ_base * 0.5)
    δ        = exp(log_δ_base * 0.5)

	ρ = length(LeafTraversal(tree)) / 1000
	sampled_model = VaryingTypeChangeRateBranchingProcess(
		λ_xscale, λ_xshift, λ_yscale, λ_yshift, μ, δ, Γ, ρ, 0, type_space, maximum(node.time for node in LeafTraversal(tree))
	)

	-loglikelihood(sampled_model, tree)
end

function main()
	treeset = load_object("treeset.jld2")
	treeset = [treeset[1]]

	type_space = load_object("type_space.jld2")
	Γ = load_object("tc_rate_matrix.jld2")
	# model = Model(treeset, Γ, type_space, maximum(node.time for node in LeafTraversal(treeset[1])))

	Random.seed!(1)

	f = x -> objective(x, treeset[1], Γ, type_space)
	grad = x -> Zygote.gradient(f, x)

	Base.length(::AbstractBranchingProcess) = 1

	optimize(f, grad, randn(6), LBFGS(); inplace=false)

end

main()
