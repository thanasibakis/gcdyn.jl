# I'm only simulating one treeset here (not a full simulations study),
# so it's okay if I can't quite recover the truth

using gcdyn, CSV, DataFrames, Optim, Random, Turing

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
			ρ = 0.5
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
			ρ = 0.5
			sampled_model = BranchingProcess(
				0, 0, 0, λ, μ, δ, Γ, ρ, 0, type_space, present_time
			)
			
			Turing.@addlogprob! loglikelihood(sampled_model, tree)
		end
	end
end

function main()
    Random.seed!(1)
    
	println("Setting up model...")

    type_space = [2, 4, 6, 8]
    Γ = [-1 0.5 0.25 0.25; 2 -4 1 1; 2 2 -5 1; 0.125 0.125 0.25 -0.5]
    present_time = 3

    truth = BranchingProcess(0, 0, 0, 2, 1.3, 1, Γ, 0.5, 0, type_space)
    treeset = rand_tree(truth, present_time, truth.type_space[1], 10)
    
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
		sampled_model = BranchingProcess(
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
		sampled_model = BranchingProcess(
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
end

main()
