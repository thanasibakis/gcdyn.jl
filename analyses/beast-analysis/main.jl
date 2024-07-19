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

function resample_tree(germinal_center_dir, discretization_table, current_parameters, beast_logdensities, Γ, type_space)
	treefiles = readdir(germinal_center_dir; join=true)[2:end] # Skip the initial tree
	weights = Vector{Float64}(undef, length(treefiles))

	Threads.@threads for i in eachindex(treefiles)
		tree = load_tree(treefiles[i], discretization_table)
		state = parse(Int, match(r"tree-STATE_(\d+)", basename(treefiles[i])).captures[1])
		present_time = maximum(node.time for node in LeafTraversal(tree))
		ρ = length(LeafTraversal(tree)) / 1000

		current_model = DiscreteBranchingProcess(
			current_parameters[1:5], current_parameters[6], current_parameters[7], Γ, ρ, 0, type_space
		)

		weights[i] = exp(
			loglikelihood(current_model, tree, present_time) - beast_logdensities[germinal_center_dir][state]
		)
	end

	load_tree(sample(treefiles, Weights(weights)), discretization_table)
end

function main()
	out_path = "out/inference/"

	# These are computed in a separate script
	type_space = [-0.47337548031309196, 0.0, 0.3890161743465148, 0.8772905477925974, 1.5050315480924654]
	discretization_table = Dict([-0.0019916536956188, 0.1112097799378943] => 0.0, [1.2169193081838328, 3.1370222155629772] => 1.5050315480924654, [0.1112097799378943, 0.6232833274714342] => 0.3890161743465148, [-5.7429821755606145, -0.0019916536956188] => -0.47337548031309196, [0.6232833274714342, 1.2169193081838328] => 0.8772905477925974)
	Γ = [-0.00907740029763729 0.005242981206393951 0.0028171242303012276 0.0009651258937143094 5.216896722780051e-5; 0.38650148502736853 -0.43123014353201944 0.0364137155775043 0.008314942927146652 0.0; 0.23740643055347962 0.03255759055032893 -0.2951301586643331 0.023230280825099564 0.0019358567354249635; 0.13084433221239158 0.018610659734128263 0.10995481618428841 -0.2861863695850132 0.026776561454204952; 0.09805142650302406 0.0024718847017569092 0.03460638582459673 0.1095868884445563 -0.24471658547393402]

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

	# Load the tree densities from BEAST

	beast_logdensities = map(germinal_center_dirs) do germinal_center_dir
		beast_logfile = filter(
			endswith(".log"),	
			readdir(joinpath("data/raw", basename(germinal_center_dir)); join=true)
		) |> first

		df = CSV.read(beast_logfile, DataFrame; header=5, select = [:state, :skyline])

		germinal_center_dir => Dict{Int, Float64}(zip(df.state, df.skyline))
	end |> Dict

	# Main SIR loop
	
	num_mcmc_iterations = 1000
	chain = Matrix{Float64}(undef, num_mcmc_iterations+1, 7)
	chain[1, :] = max_a_posteriori.values[8:14]

	for mcmc_iteration in 1:num_mcmc_iterations
		if mcmc_iteration % 10 == 0
			println("INFO: Starting SIR iteration $mcmc_iteration")
		end

		treeset = map(germinal_center_dirs) do germinal_center_dir
			resample_tree(germinal_center_dir, discretization_table, chain[mcmc_iteration, :], beast_logdensities, Γ, type_space)
		end

		parameter_samples::Chains = sample(
			DiscreteModel(treeset, Γ, type_space),
			NUTS(adtype=AutoForwardDiff(chunksize=2+length(type_space))),
			10,
			init_params=chain[mcmc_iteration, :]
		)

		chain[mcmc_iteration+1, :] = parameter_samples.value[end, 8:14, 1]
	end

	posterior_samples = DataFrame(chain, names(max_a_posteriori.values[8:14])[1])
	posterior_samples.iteration = 0:num_mcmc_iterations
	CSV.write(joinpath(out_path, "samples-posterior.csv"), posterior_samples)

	println("Done!")
end

main()
