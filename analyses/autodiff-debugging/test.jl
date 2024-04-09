println("Loading packages...")
using gcdyn, JLD2, Optim, Random, Turing

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

@model function Model2(trees, Γ, type_space, present_time)
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
			
			Turing.@addlogprob! gcdyn.stadler_appx_loglikelihood(sampled_model, tree)
		end
    end
end

function main()
	println("Loading tree...")
	treeset = load_object("treeset.jld2")

	println("Loading model...")
	type_space = load_object("type_space.jld2")
	Γ = load_object("tc_rate_matrix.jld2")
	# model = Model(treeset, Γ, type_space, maximum(node.time for node in LeafTraversal(treeset[1])))
	model2 = Model2(treeset, Γ, type_space, maximum(node.time for node in LeafTraversal(treeset[1])))

	println("Optimizing...")
	Random.seed!(1)

	for _ in 1:10
		display(optimize(model2, MLE()))
	end

	println("---------------")

	for _ in 1:10
		display(optimize(model2, MAP()))
	end

	# optimize(model, MAP(), Adam()) # Adam doesn't converge but fails nicely, unlike BFGS

	# println("Creating model...")
	# model = VaryingTypeChangeRateBranchingProcess(Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(9.462347059886547e18,7.09676029491491e18,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(491.89815525969203,0.0,1.4142135623730951,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(1.203009077565853e-67,0.0,0.0,9.022568081743897e-68,0.0,0.0,0.0), Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(0.0,0.0,0.0,0.0,0.0,0.0,0.0), Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(2.062550412105578e165,0.0,0.0,0.0,0.0,1.031275206052789e165,0.0), Dual{ForwardDiff.Tag{DynamicPPL.DynamicPPLTag, Float64}}(4.642217534714184e-77,0.0,0.0,0.0,0.0,0.0,2.321108767357092e-77), [-0.12095407527561068 0.023737729756858447 0.058638430630358486 0.028761066759347163 0.009816848129046577; 0.4596785965963223 -0.9454927655743754 0.23952483394550844 0.1598882075117643 0.08640112752078032; 0.31895943257953324 0.0648941221360519 -0.7706004779340516 0.22375383088948683 0.16299309232897963; 0.13324586619866405 0.038884487569381995 0.1932825017292408 -0.7291791366023849 0.363766281105098; 0.013095000365659267 0.006325132252092023 0.03972973695845302 0.09586528569576973 -0.15501515527197404], 0.094, 0.0, [-1.224171991580991, 0.0, 0.4104891647333908, 1.0158145231080074, 2.1957206408364116], 20.052539519990003)
	# println("Evaluating likelihood...")
	# loglikelihood(model, treeset)

	println("Done!")
end

main()
