using gcdyn, JLD2
import JSON

# Returns TreeNode{String} with types corresponding to sequences
function create_treenode(json::Vector{Dict{String, Any}})
	# For now, specify all event types to be root; we will correct this after the structure is built.
	treenodes = Dict{Int, TreeNode{String}}(node["name"] => TreeNode(:root, (isnothing(node["length"]) ? 0 : node["length"]), node["state"]) for node in json)

	for json_node in json
		if !isnothing(json_node["parent"])
			node = treenodes[json_node["name"]]
			parent = treenodes[json_node["parent"]]

			attach!(parent, node)
		end
	end

	# BEAST doesn't infer the length of the branch leading to the first birth event;
	# this means the most ancestral node is actually a birth event. We will need to
	# place a root node above this node to correct the tree structure.

	most_ancestral_nodes = filter(node -> isnothing(node.up), collect(values(treenodes)))
	@assert length(most_ancestral_nodes) == 1

	root = TreeNode(:root, 0, most_ancestral_nodes[1].type)
	attach!(root, most_ancestral_nodes[1])

	# 1. Correct `node.time` to be time since root, not branch length
	# 2. Correct the node event type
	for node in PreOrderTraversal(root.children[1])
		node.time = node.time + node.up.time
		node.event =
			if length(node.children) == 0
				:sampled_survival
			elseif length(node.children) == 1
				:type_change
			elseif length(node.children) == 2
				:birth
			else
				throw(ArgumentError("Unable to determine event type for node \$(node.name)"))
			end
	end

	# Now we must add the mutation events per the history.
	# Remember that the history describes mutation leading TO this node
	present_time = maximum(leaf.time for leaf in LeafTraversal(root))
	histories = Dict{TreeNode{String}, Vector{Dict{String, Any}}}(treenodes[node["name"]] => node["history"] for node in json)

	for node in PreOrderTraversal(root.children[1])
		history = histories[node]

		for mutation in history
			# Use a new key to invert the times so we don't modify the original,
			# otherwise this function has catastrophic side effects and can't be run twice
			mutation["when-inverted"] = present_time - mutation["when"]
		end

		sort!(history, by = mutation -> mutation["when-inverted"])

		current_node::TreeNode{String} = node.up

		for mutation in history
			index::Int = mutation["site"]
			time::Float64 = mutation["when-inverted"]
			from_base::String = mutation["from_base"]
			to_base::String = mutation["to_base"]

			@assert current_node.time < time < node.time
			@assert string(current_node.type[index]) == from_base

			detach!(current_node, node)

			new_sequence = current_node.type[1:index-1] * to_base * current_node.type[index+1:end]
			new_node = TreeNode(:type_change, time, new_sequence)

			attach!(current_node, new_node)
			attach!(new_node, node)
			
			current_node = new_node
		end
		
		if current_node.type != node.type
			if node.up.event == :root
				throw(ArgumentError("Bad tree? Root sequence may not be the naive sequence."))
			end

			@assert current_node.type == node.type
		end
	end

	# Finally, we need to make a tricky correction.
	# For the replay experiment, we tricked BEAST into "setting" the root sequence
	# by specifying a phony sampled leaf at time ε close to 0, which should in theory be the only
	# node in one of the two root-most subtrees (or there might be a couple mutations).
	# By folding this small subtree upwards, the phony node with the desired sequence will
	# be the root of the tree, and any mutations that preceded it will occur after it.

	ancestral_birth = root.children[1]
	@assert ancestral_birth.event == :birth && length(ancestral_birth.children) == 2

	phony_subtree, rest_of_tree = sort(ancestral_birth.children, by = subtree -> length(LeafTraversal(subtree)))
	@assert length(LeafTraversal(phony_subtree)) == 1
	detach!(ancestral_birth, rest_of_tree)

	# Unfold the type change events along the phony subtree, if any.
	# First, we need to go through and update the type of type_change events, since a type_change
	# event's type is supposed to be the new type, not the old type, and inverting the chain of nodes
	# won't preserve this.
	# Then, we'll go ahead and invert the chain of nodes, prepending to `rest_of_tree`.
	for node in PostOrderTraversal(phony_subtree)
		if node.event == :type_change
			node.type = node.up.type
		end
	end

	detach!(ancestral_birth, phony_subtree)

	while phony_subtree.event != :sampled_survival
		@assert phony_subtree.event == :type_change && length(phony_subtree.children) == 1

		child = phony_subtree.children[1]
		detach!(phony_subtree, child)

		phony_subtree.time = ancestral_birth.time - abs(ancestral_birth.time - phony_subtree.time)
		attach!(phony_subtree, rest_of_tree)
		rest_of_tree = phony_subtree

		phony_subtree = child
	end

	# `phony_subtree` is now the phony leaf with the naive sequence
	phony_subtree.time = ancestral_birth.time - abs(ancestral_birth.time - phony_subtree.time)
	phony_subtree.event = :root
	attach!(phony_subtree, rest_of_tree)
	root = phony_subtree

	# We technically have negative times now, so we need to update the entire tree
	root_time = root.time # Save this because we will update the root.time below
	for node in PreOrderTraversal(root)
		node.time = node.time - root_time
		@assert (node.event == :root && node.time == 0) || (node.event != :root && node.time > 0)
	end

	# Make sure we didn't make mistakes anywhere
	check_tree(root)

	return root
end

# Checks that basic tree structure is correct
function check_tree(tree::TreeNode)
	@assert tree.event == :root

	for node in PreOrderTraversal(tree)
		if node.event == :root
			@assert length(node.children) == 1
			@assert isnothing(node.up)
			@assert node.time == 0
		elseif node.event == :sampled_survival
			@assert length(node.children) == 0
			@assert !isnothing(node.up)
			@assert node.type == node.up.type
			@assert node.time > 0 && node.time > node.up.time
		elseif node.event == :type_change
			@assert length(node.children) == 1
			@assert !isnothing(node.up)
			@assert node.type != node.up.type
			@assert node.time > 0 && node.time > node.up.time
		elseif node.event == :birth
			@assert length(node.children) == 2
			@assert !isnothing(node.up)
			@assert node.type == node.up.type
			@assert node.time > 0 && node.time > node.up.time
		else
			throw(ArgumentError("Unknown event type: \$(node.event)"))
		end
	end
end

function main()
	mkpath("data/jld2-with-sequences/")
	mkpath("data/jld2-with-affinities/")

	Threads.@threads for filename in readdir("data/json/")
		basename = split(filename, ".")[1]
		mkpath("data/jld2-with-sequences/$basename/")
		mkpath("data/jld2-with-affinities/$basename/")

		json_treeset::Vector{Vector{Dict{String, Any}}} = JSON.parse(read("data/json/$filename", String))

		trees = Vector{TreeNode{String}}(undef, length(json_treeset))
		defined_i = Int[]

		for (i, json_tree) in enumerate(json_treeset)
			try
				trees[i] = create_treenode(json_tree)
				push!(defined_i, i)
			catch e
				println("Error in $basename tree $i: $e")
			end
		end

		# Currently, node.type is the sequence of the node. Let's map these to affinities
		sequences = unique(node.type for tree in trees[defined_i] for node in PreOrderTraversal(tree))
		affinities = pipeline(`bin/get-affinity`; stdin=IOBuffer(join(sequences, "\n"))) |>
			(command -> read(command, String)) |>
			strip |>
			(text -> split(text, "\n")) |>
			(lines -> parse.(Float64, lines))

		affinity_map = Dict(sequence => affinity for (sequence, affinity) in zip(sequences, affinities))

		for i in defined_i
			save_object("data/jld2-with-sequences/$basename/tree-$i.jld2", trees[i])
			tree_with_affinities::TreeNode{Float64} = map_types(type -> affinity_map[type], trees[i])
			save_object("data/jld2-with-affinities/$basename/tree-$i.jld2", tree_with_affinities)
		end
	end
end

main()
