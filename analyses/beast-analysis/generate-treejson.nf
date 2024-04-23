workflow {
    // Workflow used to generate the JSON files, if not already present

    // Create tuples of the form (id, nexus_file)
    def history_treesfile = Channel.fromPath("data/raw/*/*.history.trees") | map { [it.simpleName, it] }
    def treejson_file = history_treesfile | beast_to_treejson
}

process beast_to_treejson {
    publishDir "data/json/", saveAs: { "${id}." + it }, mode: "copy", overwrite: false

    input:
    tuple val(id), path(history_treesfile)

    output:
    tuple val(id), path("trees.json")

    """
    # Extract the NEXUS header
    grep -v "^tree" $history_treesfile | sed "\$d" > temp.history.trees
    
	beast-to-treejson --compact-history $history_treesfile > trees.json
	"""
}
