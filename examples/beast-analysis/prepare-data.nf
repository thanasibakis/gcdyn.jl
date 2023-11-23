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

    # Keep the last several trees
    # tail -n 11 $history_treesfile >> temp.history.trees

    # beast-to-treejson --compact-history temp.history.trees > trees.json
    
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
	treejson-to-julia $treejson_file
    """
}
