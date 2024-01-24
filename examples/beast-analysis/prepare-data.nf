workflow {
    // Create tuples of the form (id, nexus_file)
    def history_treesfile = Channel.fromPath("data/*/*.history.trees") | map { [it.simpleName, it] }

    history_treesfile |
        beast_to_treejson |
		treejson_to_julia
}

process beast_to_treejson {
    publishDir "output/json/", saveAs: { "${id}." + it }

    input:
    tuple val(id), path(history_treesfile)

    output:
    tuple val(id), path("trees.json")

    """
    # Extract the NEXUS header
    grep -v "^tree" $history_treesfile | sed '\$d' > temp.history.trees
    
	beast-to-treejson --compact-history $history_treesfile > trees.json
	"""
}

process treejson_to_julia {
    publishDir "output/jld2/", saveAs: { "${id}." + it }

    input:
    tuple val(id), path(treejson_file)

    output:
    tuple val(id), path("trees.jld2")

    """
    #!/usr/bin/env julia

    # Run this program as though it were invoked with `julia --project`
    # to find the project in the `examples` directory
    pushfirst!(LOAD_PATH, "@.")
    import Pkg; Pkg.activate()

    using gcdyn, JLD2
    import JSON

    # Shorthand for finding the node matching a given name
    get_node(json_tree, node_name) = filter(d -> d["name"] == node_name, json_tree)[1]

    # Create the tree structure.
    # We will need to fill in the states later
    function create_treenode(json_tree)
        # For now, specify the event type to be root, but we will correct this after the structure is built.
        # Also for now, specify the state to be -1, but this will be populated after *all* trees are built.
        nodes = Dict(node["name"] => TreeNode(node["name"], :root, (isnothing(node["length"]) ? 0 : node["length"]), -1, []) for node in json_tree)

        for json_node in json_tree
            if !isnothing(json_node["parent"])
                node = nodes[json_node["name"]]
                parent = nodes[json_node["parent"]]

                push!(parent.children, node)
                node.up = parent
            end
        end

        # BEAST doesn't infer the length of the branch leading to the first birth event;
        # this means `tree` is really a tuple of the two root-most subtrees.
        # For the replay experiment, though, we trick BEAST into "knowing" the root sequence
        # by inducing a birth event very close to time 0, so that one of the root-most subtrees
        # is the root sequence. We don't include that in the final tree though.
        # Thus, here, we just need to add a root event node before the other subtree.
        subtrees = filter(node -> isnothing(node.up), collect(values(nodes)))[1].children
        first_child = (length(LeafTraversal(subtrees[1])) == 1) ? subtrees[2] : subtrees[1]
        
        root = TreeNode(0, :root, 0, -1, [first_child])

        # 1. Correct `node.t` to be time since root, not branch length
        # 2. Fill in the sequences (into temporary storage, since the node state will be the corresponding affinity)
        # 3. Correct the node event type
        for node in PreOrderTraversal(root.children[1])
            node.t = node.t + node.up.t
            node.info[:sequence] = get_node(json_tree, node.name)["state"]
            node.event =
                if length(node.children) == 0
                    :sampled_survival
                elseif length(node.children) == 1
                    :mutation
                elseif length(node.children) == 2
                    :birth
                else
                    throw(ArgumentError("Unable to determine event type for node $(node.name)"))
                end
        end

        root_sequence = (length(LeafTraversal(subtrees[1])) == 1) ? get_node(json_tree, subtrees[1].name)["state"] : get_node(json_tree, subtrees[2].name)["state"]
        root.info[:sequence] = root_sequence

        root
    end

    # Add mutation events per the history
    # Remember that the history describes mutation leading TO this node
    function expand_mutation_history!(root, json_tree)
        present_time = maximum(leaf.t for leaf in LeafTraversal(root))

        for node in PreOrderTraversal(root.children[1])
            history = get_node(json_tree, node.name)["history"]

            for mutation in history
                mutation["when"] = present_time - mutation["when"]
            end

            sort!(history, by = mutation -> mutation["when"], rev = true)

            current_node = node

            for mutation in history
                index = mutation["site"]
                current_sequence = current_node.info[:sequence]

                mutation_time = mutation["when"]

                @assert current_node.up.t < mutation_time < current_node.t
                @assert string(current_sequence[index]) == mutation["to_base"]

                new_sequence = current_sequence[1:index-1] * mutation["from_base"] * current_sequence[index+1:end]

                parent = current_node.up
                filter!(child -> child != current_node, parent.children)
                # Removing current_node.up is done in the TreeNode constructor in the next step

                new_node = TreeNode(node.name, :mutation, mutation["when"], -1, [current_node])
                new_node.info[:sequence] = new_sequence

                push!(parent.children, new_node)
                new_node.up = parent

                current_node = new_node
            end
        end
    end

    # Compute the affinities for all sequences in one go, since it's fastest if we call the binary once with a batch
    function predict_affinities!(trees)
        all_nodes = [node for tree in trees for node in PreOrderTraversal(tree)]
        sequences = [node.info[:sequence] for node in all_nodes]

        affinities = pipeline(`get-affinity`; stdin=IOBuffer(join(sequences, "\n"))) |>
            (command -> read(command, String)) |>
            strip |>
            (text -> split(text, "\n")) |>
            (lines -> parse.(Float64, lines))

        for (affinity, node) in zip(affinities, all_nodes)
            node.state = affinity
        end
    end

    function main()
        json_trees = JSON.parse(read("$treejson_file", String))

        trees = map(json_trees) do json_tree
            root = create_treenode(json_tree)
            expand_mutation_history!(root, json_tree)

            root
        end

        predict_affinities!(trees)

        save_object("trees.jld2", trees)
    end

    main()
    """
}
