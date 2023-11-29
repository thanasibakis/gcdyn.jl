workflow {
	//trees = simulate_trees | import_trees_to_julia
	trees = import_trees_to_julia(file("trees.json"))
	//run_ids = Channel.from(1..100)
	run_ids = Channel.from(1..3)
	
	inference(trees, run_ids) | collect | merge_samples
}

process simulate_trees {
    publishDir "output/"

    output:
    path("trees.json")

    """
	#!/usr/bin/env python3

	import json
	from pathlib import Path

	from gcdyn import bdms, gpmap, mutators, poisson
	from experiments import replay

	birth_response = poisson.SigmoidResponse(xscale=1.0, xshift=1.0, yscale=5.0, yshift=0.1)
	death_response = poisson.ConstantResponse(1.0)
	mutation_response = poisson.SequenceContextMutationResponse(replay.mutability(), mutation_intensity=1.0)

	dms = replay.dms(Path("support") / "final_variant_scores.csv")
	gp_map = gpmap.AdditiveGPMap(dms["affinity"], nonsense_phenotype=dms["affinity"].min().min())
	mutator = mutators.SequencePhenotypeMutator(
		mutators.ContextMutator(
			mutability=replay.mutability(),
			substitution=replay.substitution()
		),
		gp_map
	)

	trees = []

	def simulate_tree():
		tree = bdms.TreeNode()
		tree.x = gp_map(replay.NAIVE_SEQUENCE)
		tree.sequence = replay.NAIVE_SEQUENCE
		tree.sequence_context = replay.seq_to_contexts(replay.NAIVE_SEQUENCE)

		for iter in range(1000):
			try:
				tree.evolve(
					t = 20,
					birth_response=birth_response,
					death_response=death_response,
					mutation_response=mutation_response,
					mutator=mutator,
					capacity=100,
					init_population=10,
					min_survivors=50,
					capacity_method="death",
					birth_mutations=False,
					verbose=False,
				)

				tree.sample_survivors(n=50)
				tree.prune()
				
				return tree
			except bdms.TreeError as e:
				print(f"try {iter + 1} failed, {e}", flush=True)
				continue
		
		raise bdms.TreeError("1000 attempts failed")

	def node_to_json(node):
		return {
			"name": node.name,
			"parent": node.up.name if node.up else None,
			"length": node.t - node.up.t if node.up else None,
			"state": node.x
		}

	json_trees = []

	for _ in range(67*100):
		tree = simulate_tree()
		json_tree = [node_to_json(node) for node in tree.traverse()]
		json_trees.append(json_tree)

	with open("trees.json", "w") as file:
		json.dump(json_trees, file)
	"""
}

process import_trees_to_julia {
	publishDir "output/"

	input:
	path(treejson_file)

	output:
	path("trees.jld2")

	"""
	#!/usr/bin/env julia

	# Run this script as though it were invoked with `julia --project`
	# to find the project in the `examples` directory
	pushfirst!(LOAD_PATH, "@.")
	import Pkg; Pkg.activate()

	using gcdyn, CategoricalArrays, DataFrames, Distributions, JLD2, JSON, StatsPlots

	function create_treenode(json_tree)
		# For now, specify the event type to be root, but we will correct this after the structure is built.
		nodes = Dict(node["name"] => TreeNode(node["name"], :root, (isnothing(node["length"]) ? 0 : node["length"]), node["state"], []) for node in json_tree)

		for json_node in json_tree
			if !isnothing(json_node["parent"])
				node = nodes[json_node["name"]]
				parent = nodes[json_node["parent"]]

				push!(parent.children, node)
				node.up = parent
			end
		end

		root = filter(node -> isnothing(node.up), collect(values(nodes)))[1]

		# 1. Correct `node.t` to be time since root, not branch length
		# 2. Correct the node event type
		for node in PreOrderTraversal(root.children[1])
			node.t = node.t + node.up.t
			node.event =
				if length(node.children) == 0
					:sampled_survival
				elseif length(node.children) == 1
					:mutation
				elseif length(node.children) == 2
					:birth
				else
					throw(ArgumentError("Unable to determine event type for node \$(node.name)"))
				end
		end

		root
	end

	function bin_states!(trees)
		all_nodes = [node for tree in trees for node in PostOrderTraversal(tree)]

		# Create bins from evenly spaced quantiles, then discretize states to the medians of their bins
		states = DataFrame(state=[node.state for node in all_nodes])
		num_bins = 5
		cutoffs = quantile(states.state, 0:(1/num_bins):1)
		states.bin = cut(states.state, cutoffs; extend=true)
		binned_states = transform(groupby(states, :bin), :state => median => :binned_state).binned_state
		
		for (node, binned_state) in zip(all_nodes, binned_states)
			node.state = binned_state
		end

		# Visualize
		ENV["GKSwstype"] = "100" # operate plotting in headless mode
		histogram(states.state; normalize=:pdf, fill="grey", alpha=0.2)
		histogram!(binned_states; normalize=:pdf)
		png("binned-states.png")

		nothing
	end

	function prune_same_bin_mutations!(tree)
		# If a mutation resulted in an state of the same bin as the parent,
		# that isn't a valid type change in the CTMC, so we prune it

		for node in PostOrderTraversal(tree)
			if node.event == :mutation && node.state == node.up.state
				filter!(child -> child != node, node.up.children)
				push!(node.up.children, node.children[1])
				node.children[1].up = node.up
			end
		end
	end

	json_trees = JSON.parse(read("$treejson_file", String))
	trees = map(create_treenode, json_trees)
	bin_states!(trees)
	map(prune_same_bin_mutations!, trees)

	save_object("trees.jld2", trees)
	"""
}

process inference {
	publishDir "output/"

	input:
	path("trees.jld2")
	val(run_id)

	output:
	path("posterior-samples.csv")

	"""
	#!/usr/bin/env julia

	# Run this script as though it were invoked with `julia --project`
	# to find the project in the `examples` directory
	pushfirst!(LOAD_PATH, "@.")
	import Pkg; Pkg.activate()

	println("Loading packages...")
	using gcdyn, CategoricalArrays, CSV, DataFrames, Distributions, JLD2, JSON, StatsPlots, Turing

	function main()
		trees = load_object("trees.jld2")

		state_space = unique(node.state for tree in trees for node in PostOrderTraversal(tree))
		present_time = maximum(leaf.t for tree in trees for leaf in LeafTraversal(tree))

		println("Sampling from prior...")
		prior_samples = sample(Model(missing, state_space, present_time), Prior(), 100) |> DataFrame
		prior_samples.p = gcdyn.expit.(prior_samples[:, :logit_p])
		prior_samples.δ = gcdyn.expit.(prior_samples[:, :logit_δ])

		println("Sampling from posterior...")
		num_trees = 67
		treeset = trees[(1:num_trees) .+ num_trees * ($run_id - 1)]

		posterior_samples = sample(
			Model(treeset, state_space, present_time),
			MH(
				:xscale => x -> LogNormal(log(x), 0.2),
				:xshift => x -> Normal(x, 0.7),
				:yscale => x -> LogNormal(log(x), 0.2),
				:yshift => x -> LogNormal(log(x), 0.2),
				:μ => x -> LogNormal(log(x), 0.2),
				:γ => x -> LogNormal(log(x), 0.2),
				:logit_p => x -> Normal(x, 0.2),
				:logit_δ => x -> Normal(x, 0.2)
			),
			5000
		) |> DataFrame

		posterior_samples.run .= $run_id

		println("Exporting samples...")
		CSV.write("posterior-samples.csv", posterior_samples)
	end

	@model function Model(trees, state_space, present_time)
		xscale  ~ Gamma(2, 1)
		xshift  ~ Normal(5, 1)
		yscale  ~ Gamma(2, 1)
		yshift  ~ Gamma(1, 1)
		μ       ~ LogNormal(0, 0.5)
		γ       ~ LogNormal(0, 0.5)
		logit_p ~ Normal(0, 1.8)
		logit_δ ~ Normal(1.4, 1.4)

		p = gcdyn.expit(logit_p)  # roughly Uniform(0, 1)
		δ = gcdyn.expit(logit_p)  # roughly Beta(3, 1)

		sampled_model = SigmoidalBirthRateBranchingProcess(
			xscale, xshift, yscale, yshift, μ, γ, state_space, random_walk_transition_matrix(state_space, p; δ=δ), 1, 0, present_time
		)

		# Only compute loglikelihood if we're not sampling from the prior
		if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
			Turing.@addlogprob! loglikelihood(sampled_model, trees; reltol=1e-3, abstol=1e-3)
		end
	end

	main()
	"""
}

process merge_samples {
	publishDir "output/"

	input:
	path("posterior-samples-*.csv")

	output:
	path("posterior-samples.csv")

	"""
	# combine the csv files and keep only one header row

	# get the header row
	head -n 1 posterior-samples-1.csv > posterior-samples.csv

	# append all the other rows
	tail -n +2 -q posterior-samples-*.csv >> posterior-samples.csv
	"""
}
