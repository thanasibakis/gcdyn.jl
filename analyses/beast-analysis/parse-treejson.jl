using gcdyn, JLD2
import JSON

# Returns TreeNode{String} with types corresponding to sequences
function create_treenode(json)
	# For now, specify all event types to be root; we will correct this after the structure is built.
	treenodes = Dict(node["name"] => TreeNode(:root, (isnothing(node["length"]) ? 0 : node["length"]), node["state"]) for node in json)

	for json_node in json
		if !isnothing(json_node["parent"])
			node = treenodes[json_node["name"]]
			parent = treenodes[json_node["parent"]]

			attach!(parent, node)
		end
	end

	histories = Dict(treenodes[node["name"]] => node["history"] for node in json)

	# BEAST doesn't infer the length of the branch leading to the first birth event;
	# this means `tree` is really a tuple of the two root-most subtrees.
	# For the replay experiment, though, we trick BEAST into "setting" the root sequence
	# by specifying a phony sampled leaf at time ε≈0, which should in theory be the only
	# node in one of the two root-most subtrees. By ignoring this node and placing a root
	# at the start of the other subtree with the phony node's sequence, we can then have
	# a tree with our desired root sequence.
	subtrees = filter(node -> isnothing(node.up), collect(values(treenodes)))[1].children
	first_child = (length(LeafTraversal(subtrees[1])) == 1) ? subtrees[2] : subtrees[1]
	phony_node = (length(LeafTraversal(subtrees[1])) == 1) ? subtrees[1] : subtrees[2]
	detach!(first_child.up, first_child)
	
	root = TreeNode(:root, 0, phony_node.type)
	attach!(root, first_child)

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

	for node in PreOrderTraversal(root.children[1])
		history = histories[node]

		for mutation in history
			# Use a new key to invert the times so we don't modify the original,
			# otherwise this function has catastrophic side effects and can't be run twice
			mutation["when-inverted"] = present_time - mutation["when"]
		end

		sort!(history, by = mutation -> mutation["when-inverted"])

		current_node = node.up

		for mutation in history
			index = mutation["site"]

			@assert current_node.time < mutation["when-inverted"] < node.time
			@assert string(current_node.type[index]) == mutation["from_base"]

			detach!(current_node, node)

			new_sequence = current_node.type[1:index-1] * mutation["to_base"] * current_node.type[index+1:end]
			new_node = TreeNode(:type_change, mutation["when-inverted"], new_sequence)

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

	return root
end

function main()
	mkpath("data/jld2")

	Threads.@threads for filename in readdir("data/json/")
		basename = split(filename, ".")[1]
		mkpath("data/jld2/$basename")

		json_treeset = JSON.parse(read("data/json/$filename", String))

		for (i, json_tree) in enumerate(json_treeset)
			try
				tree = create_treenode(json_tree)
				save_object("data/jld2/$basename/tree-$i.jld2", tree)
			catch e
				println("Error in $basename tree $i: $e")
			end
		end
	end
end

main()
