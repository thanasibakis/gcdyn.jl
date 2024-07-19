println("Loading packages...")
using CSV, gcdyn, DataFrames, Distributions, JLD2, LinearAlgebra, Optim, StatsBase, Turing

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

function load_tree(path, discretization_table)
	tree = load_object(path)::TreeNode{Float64}

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

function main()
	out_path = "out/inference/"

	# These are computed in a separate script
	type_space = [-0.47337548031309196, 0.0, 0.3890161743465148, 0.8772905477925974, 1.5050315480924654]
	discretization_table = Dict([-0.0019916536956188, 0.1112097799378943] => 0.0, [1.2169193081838328, 3.1370222155629772] => 1.5050315480924654, [0.1112097799378943, 0.6232833274714342] => 0.3890161743465148, [-5.7429821755606145, -0.0019916536956188] => -0.47337548031309196, [0.6232833274714342, 1.2169193081838328] => 0.8772905477925974)
	Γ = [-0.13920078945911146 0.0189548320216946 0.06020245390538222 0.039379954996853976 0.020663548535180695; 0.5163595766457914 -0.967454374087282 0.1905155687066721 0.1607625075988291 0.09981672113598941; 0.36330813246625304 0.040048301304979946 -0.7864193336973415 0.19467425387712223 0.18838864604898636; 0.17596705572231539 0.03200893271068237 0.17219164314618363 -0.774452023225638 0.3942843916464567; 0.022957787990039946 0.006690906125439449 0.04382850434462171 0.09017377154376652 -0.16365097000386764]
	
	# Read in the trees used to compute the MAP
	println("Reading trees...")
	germinal_center_dirs = readdir("data/jld2-with-affinities/"; join=true)

	starter_treeset = map(germinal_center_dirs) do germinal_center_dir
		load_tree(joinpath(germinal_center_dir, "tree-STATE_10000000.jld2"), discretization_table)
	end

	println("Sampling from prior...")
	prior_samples = sample(DiscreteModel([], Γ, type_space), Prior(), 5000) |> DataFrame
	mkpath(out_path)
	CSV.write(joinpath(out_path, "samples-prior.csv"), prior_samples)

	println("Computing initial MCMC state...")
	max_a_posteriori = optimize(DiscreteModel(starter_treeset, Γ, type_space), MAP())

	open(joinpath(out_path, "map.txt"), "w") do f
		println(f, max_a_posteriori)
	end

	println("Sampling from posterior...")

	# Load the tree likelihoods from BEAST

	beast_loglikelihoods = map(germinal_center_dirs) do germinal_center_dir
		beast_logfile = filter(
			endswith(".log"),	
			readdir(joinpath("data/raw", basename(germinal_center_dir)); join=true)
		) |> first

		df = CSV.read(beast_logfile, DataFrame; header=5, select = [:state, :likelihood])
		states, loglikelihoods = df.state::AbstractVector{Int}, df.likelihood::AbstractVector{Float64}

		germinal_center_dir => (state -> loglikelihoods[findfirst(==(state), states)])
	end |> Dict

	# Main SIR loop
	num_mcmc_iterations = 1000
	chain = Matrix{Float64}(undef, num_mcmc_iterations+1, 7)
	chain[1, :] = max_a_posteriori.values[8:14]

	for mcmc_iteration in 1:num_mcmc_iterations
		treeset = map(germinal_center_dirs) do germinal_center_dir
			treefiles = readdir(germinal_center_dir; join=true)[2:end] # Skip the initial tree
			weights = Vector{Float64}(undef, length(treefiles))

			Threads.@threads for i in eachindex(treefiles)
				tree = load_tree(treefiles[i], discretization_table)
				state = parse(Int, match(r"tree-STATE_(\d+)", basename(treefiles[i])).captures[1])
				present_time = maximum(node.time for node in LeafTraversal(tree))
				ρ = length(LeafTraversal(tree)) / 1000

				current_model = DiscreteBranchingProcess(
					chain[mcmc_iteration, 1:5], chain[mcmc_iteration, 6], chain[mcmc_iteration, 7], Γ, ρ, 0, type_space
				)

				# TODO: these all come out to infinity because the beast likelihoods are so small
				weights[i] = exp(loglikelihood(current_model, tree, present_time) - beast_loglikelihoods[germinal_center_dir](state))
			end

			load_tree(sample(treefiles, Weights(weights)), discretization_table)
		end

		parameter_samples = sample(
			DiscreteModel(treeset, Γ, type_space),
			NUTS(adtype=AutoForwardDiff(chunksize=2+length(type_space))),
			10,
			init_params=current_parameters
		)

		chain[mcmc_iteration+1, :] = parameter_samples.value[end, 8:14, 1]
	end

	posterior_samples = DataFrame(chain, names(max_a_posteriori.values[8:14])[1])
	posterior_samples.iteration = 0:num_mcmc_iterations
	CSV.write(joinpath(out_path, "samples-posterior.csv"), posterior_samples)

	println("Done!")
end

main()
