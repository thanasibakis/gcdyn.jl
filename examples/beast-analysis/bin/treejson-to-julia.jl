#!/usr/bin/env julia

using AbstractTrees, gcdyn, JLD2
import JSON

# Shorthand for extracting node data from the JSON `nodes` objects
get_node_data(json_tree, node_name, field) = filter(d -> d["name"] == node_name, json_tree["nodes"])[1][field]

# Methods named `↓` are used to parse Newick into TreeNode objects.
# Inspired by https://www.jamesporter.me/2013/11/27/how-to-succeed-at-parsing.html
# Requires that node labels exist only on leaves and are integers

# Create a leaf node. We will have to fill in the state and adjust `t` to be `length + up.length`
↓(name::Int, length::Real) = TreeNode(name, :sampled_survival, length, -1, [])

# Create an ancestral node. We will have to fill in the name and state and adjust `t`
↓(children::Tuple{TreeNode, TreeNode}, length::Real) = TreeNode(-1, :birth, length, -1, collect(children))

function parse_treejson(json_trees)
    trees = map(json_trees) do json_tree
        # Fancy trick to parse Newick directly into TreeNode objects
        tree = replace(json_tree["newick"], ":" => "↓") |> Meta.parse |> eval

        # Why am I missing the birth event of the single ancestor?
        # and what is its branch length? (currently i'll just put 0)
        tree = TreeNode(-1, :birth, 0, -1, collect(tree))

        # Create root node
        # TODO: document why we do this (ie. because we want to store the root state, especially if we will add a mutation to the first branch)
        root = TreeNode(0, :root, 0, tree.state, [tree])

        # Fill in the names of ancestral nodes
        for node in PostOrderTraversal(tree)
            if node.up.event != :root
                node.up.name = get_node_data(json_tree, node.name, "parent")
            end
        end

        # Correct `node.t` to be time since root, not branch length,
        # and fill in the sequences (into temporary storage, since the node state will be the corresponding affinity)
        for node in PreOrderTraversal(tree)
            node.t = node.t + node.up.t
            node.info[:sequence] = get_node_data(json_tree, node.name, "state")
        end

        root.info[:sequence] = tree.info[:sequence]

        # Add mutation events per the history
        # Remember that the history describes mutation leading TO this node
        present_time = maximum(leaf.t for leaf in Leaves(tree))

        for node in PreOrderTraversal(tree)
            history = get_node_data(json_tree, node.name, "history")

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

        root
    end

    # Compute the affinities for all sequences now, since it's fastest if we call the binary once with a batch
    all_nodes = [node for tree in trees for node in PreOrderTraversal(tree)]
    sequences = [node.info[:sequence] for node in all_nodes]

    affinities = pipeline(`bin/get-affinity`; stdin=IOBuffer(join(sequences, "\n"))) |>
        (command -> read(command, String)) |>
        strip |>
        (text -> split(text, "\n")) |>
        (lines -> parse.(Float64, lines))

    for (affinity, node) in zip(affinities, all_nodes)
        node.state = affinity
    end

    return trees
end

json_trees = isempty(ARGS) ? JSON.parse(stdin) : JSON.parse(read(ARGS[1], String))
trees = parse_treejson(json_trees)
save_object("output/tenseqs.trees.jld2", trees)
