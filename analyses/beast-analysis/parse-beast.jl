println("Loading packages...")
using gcdyn, JLD2

function main()
	println("Starting parse loop...")

	mkpath("data/jld2-with-sequences/")
	mkpath("data/jld2-with-affinities/")

	Threads.@threads for germinal_center_name in readdir("data/raw/")
		if germinal_center_name == ".DS_Store"
			continue
		end

		mkpath("data/jld2-with-sequences/$germinal_center_name/")
		mkpath("data/jld2-with-affinities/$germinal_center_name/")

		println("$germinal_center_name\t Reading BEAST trees...")

		history_trees_file = filter(readdir("data/raw/$germinal_center_name"; join=true)) do filename
			endswith(filename, ".history.trees")
		end[1]

		trees = Dict{String, TreeNode{String}}()

		open(history_trees_file) do file
			line = readline(file)

			while !startswith(line, "tree STATE_") && !eof(file)
				line = readline(file)
			end

			println("$germinal_center_name\t Exporting sequence-level trees...")

			while !startswith(line, "End;") || !eof(file)
				name = match(r"tree (?<name>STATE_\d+)", line)[:name]
				newick = split(line, " = [&R] ")[2]

				try
					trees[name] = TreeNode(newick)
					trees[name] = pivot_phony_subtree!(trees[name])
					correct_present_time!(trees[name])
					save_object("data/jld2-with-sequences/$germinal_center_name/tree-$name.jld2", trees[name])
				catch e
					println("$germinal_center_name\t ERROR for tree $name: $e")
				end

				line = readline(file)
			end
		end

		println("$germinal_center_name\t Computing affinities...")

		# Currently, node.type is the sequence of the node. Let's map these to affinities
		sequences = unique(node.type for tree in values(trees) for node in PreOrderTraversal(tree))
		affinities = pipeline(`bin/get-affinity`; stdin=IOBuffer(join(sequences, "\n"))) |>
			(command -> read(command, String)) |>
			strip |>
			(text -> split(text, "\n")) |>
			(lines -> parse.(Float64, lines))

		affinity_map = Dict(sequence => affinity for (sequence, affinity) in zip(sequences, affinities))

		println("$germinal_center_name\t Exporting affinity-level trees...")

		# Note that we're not pruning affinity-preserving mutations here. Thus, "type changes" are still on the nucleotide level,
		# even though the tree type is now the affinity. This is for visualization purposes, and `validate_tree`/loglikelihoods will fail
		# without later pruning.
		for (name, tree) in trees
			tree_with_affinities::TreeNode{Float64} = map_types(type -> affinity_map[type], tree; prune_self_loops=false)
			save_object("data/jld2-with-affinities/$germinal_center_name/tree-$name.jld2", tree_with_affinities)
		end

		println("$germinal_center_name\t Done.")
	end

	println("Parse loop complete. All done.")
end

# For the replay experiment, we tricked BEAST into "setting" the root sequence
# by specifying a phony sampled leaf at time ε close to 0, which should in theory be the only
# node in one of the two root-most subtrees (or there might be a couple mutations).
# By folding this small subtree upwards, the phony node with the desired sequence will
# be the root of the tree, and any mutations that preceded it will occur after it.
#
# Will modify the tree in place. The new root (the phony leaf) will be returned.
function pivot_phony_subtree!(root::TreeNode)
	ancestral_birth = root.children[1]
	@assert ancestral_birth.event == :birth && length(ancestral_birth.children) == 2

	# Remove the old root. We're going to return a new one.
	detach!(root, ancestral_birth)

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
	validate_tree(root)

	return root
end

# For our experimental trees, we know that all leaves were sampled at present time.
# The leaf times in each tree were determined by summing up branch lengths, so there
# might be minor floating point error for what should be the same time. We fix this here
function correct_present_time!(tree::TreeNode)
	present_time = maximum(node.time for node in LeafTraversal(tree))

	for leaf in LeafTraversal(tree)
		@assert leaf.time ≈ present_time atol=1e-5
		leaf.time = present_time
	end
end

# Checks that basic tree structure is correct.
function validate_tree(tree::TreeNode)
	@assert tree.event == :root

	num_roots_found = 0

	for node in PreOrderTraversal(tree)
		if node.event == :root
			@assert length(node.children) == 1
			@assert isnothing(node.up)
			@assert node.time == 0
			num_roots_found += 1
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

			if node.up.event == :root
				# This may be a BEAST tree, which doesn't infer a time to first birth event,
				# so the time to the first birth event is 0
				@assert node.time == 0 || (node.time > 0 && node.time > node.up.time)
			else
				@assert node.time > 0 && node.time > node.up.time
			end
		else
			throw(ArgumentError("Unknown event type: \$(node.event)"))
		end
	end

	@assert num_roots_found == 1
end

# Returns TreeNode{String} with types corresponding to sequences.
# `newick` will need to be `split(treefile_line, " = [&R] ")[2]`
# where `treefile_line` is a line from a BEAST .history.trees file, stripped of `\n`.
#
# Should also work if the tree isn't from BEAST and has a time to first birth event (which BEAST doesn't infer).
# You will still need to set a state attribute in the Newick string, though. (History is optional.)
#
# Some examples to try out:
#
# newick = "(([&states=\"A\"]:0.1,[&states=\"G\"]:0.2)[&states=\"C\"]:0.3,([&states=\"G\"]:0.4,[&states=\"T\"]:0.5)[&states=\"C\"]:0.6)[&states=\"A\"]:1.0;"
# newick = "(([&states=\"A\"]:0.1,[&states=\"G\"]:0.2)[&states=\"C\"]:0.3,([&states=\"G\"]:0.4,[&states=\"T\"]:0.5)[&states=\"C\"]:0.6)[&states=\"A\"];"
function gcdyn.TreeNode(newick)
	label_regex = r"\d*\[&states=\"(?<state>[ACGT]+)\"\]:(?:\[&history_all=\{(?<history>(?:\{\d+,\d+\.\d+(?:E-\d+)?,[ACGT],[ACGT]\}(?:,)?)+)\}])?(?<length>\d+\.\d+(?:E-\d+)?)"

	current_node = TreeNode(:birth, 0, "") # The first birth event in the tree. Info to be filled in later
	ancestors = [] # Treat this as a stack, pushing and popping off the end
	current_label = ""
	currently_inside_square_bracket = false

	histories = Dict()

	for char in newick
		if char == '('
			# Treat the current node as a parent and prepare to parse a child node.
			# The parent's info will be filled in later

			push!(ancestors, current_node)
			current_node = TreeNode(:birth, 0, "") # to be filled in later

		elseif char == ',' && !currently_inside_square_bracket
			# We have finished parsing a child node, and up next will be a sibling

			node_info = match(label_regex, current_label)
			current_node.time = parse(Float64, node_info[:length])
			current_node.type = node_info[:state]
			histories[current_node] = isnothing(node_info[:history]) ? "" : node_info[:history]

			attach!(ancestors[end], current_node)

			current_node = TreeNode(:birth, 0, "") # to be filled in later
			current_label = ""

		elseif char == ')'
			# We have finished parsing a child node, and there are no more children.
			# Up next will be node info for the parent

			node_info = match(label_regex, current_label)

			if isnothing(node_info)
				println(current_label)
			end
			current_node.time = parse(Float64, node_info[:length])
			current_node.type = node_info[:state]
			histories[current_node] = isnothing(node_info[:history]) ? "" : node_info[:history]

			attach!(ancestors[end], current_node)

			current_node = pop!(ancestors)
			current_label = ""

		elseif char == ';'
			# End of the tree. The node info we have is for the first birth event.
			# Here's the catch. BEAST trees do not infer a time to first birth event.
			# So we'll have to set that to 0 and use a regex that doesn't expect this information

			node_info = match(label_regex, current_label)

			if !isnothing(node_info)
				current_node.time = parse(Float64, node_info[:length])
				current_node.type = node_info[:state]
				histories[current_node] = isnothing(node_info[:history]) ? "" : node_info[:history]
			else
				# It's a BEAST tree
				node_info = match(r"\[&states=\"(?<state>[ACGT]+)\"\]", current_label)
				current_node.time = 0
				current_node.type = node_info[:state]
				histories[current_node] = ""
			end

			break
		else
			# We are parsing a node
			current_label *= char

			if char == '['
				currently_inside_square_bracket = true
			elseif char == ']'
				currently_inside_square_bracket = false
			end
		end
	end

	@assert isempty(ancestors)

	root = TreeNode(:root, 0, current_node.type)
	attach!(root, current_node)

	# 1. Correct `node.time` to be time since root, not branch length
	# 2. Correct the node event type
	for node in PreOrderTraversal(root.children[1])
		node.time += node.up.time
		node.event =
			if length(node.children) == 0
				:sampled_survival
			elseif length(node.children) == 2
				:birth
			else
				# BEAST should not be outputting nodes with 1 child; the history contains those
				println(node, node.children)
				throw(ArgumentError("Unable to determine event type $(node.type)"))
			end
	end

	# Finally, let's add the mutation events per the history.
	# First, we parse the history strings into a more usable format
	history_regex = r"\{(?<site>\d+),(?<when>[\d.E-]+),(?<from_base>\w+),(?<to_base>\w+)\}"
	present_time = maximum(leaf.time for leaf in LeafTraversal(root))

	for (node, history_string) in histories
		histories[node] = Dict[]
		for mutation in eachmatch(history_regex, history_string)
			push!(
				histories[node],
				Dict(
					:site => parse(Int, mutation[:site]),
					:when => present_time - parse(Float64, mutation[:when]), # BEAST encodes them as time before present
					:from_base => mutation[:from_base],
					:to_base => mutation[:to_base]
				)
			)
		end
	end

	# Remember that the history describes mutation leading TO this node
	for node in PreOrderTraversal(root.children[1])
		history = sort(histories[node], by = mutation -> mutation[:when])

		current_node::TreeNode{String} = node.up

		for mutation in history
			index = mutation[:site]

			@assert current_node.time < mutation[:when] < node.time
			@assert string(current_node.type[index]) == mutation[:from_base]

			detach!(current_node, node)

			new_sequence = current_node.type[1:index-1] * mutation[:to_base] * current_node.type[index+1:end]
			new_node = TreeNode(:type_change, mutation[:when], new_sequence)

			attach!(current_node, new_node)
			attach!(new_node, node)

			current_node = new_node
		end

		@assert current_node.type == node.type
	end

	# Make sure we didn't make mistakes anywhere
	validate_tree(root)

	return root
end

main()