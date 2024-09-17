# Run this to extract tree affinities for histogram visualization in R

using gcdyn, JLD2

trees = load_object("trees.jld2")

affinities = [node.type for tree in treees for node in PostOrderTraversal(tree)]

open("affinities.txt", "w") do f
	for a in affinities
		println(f, a)
	end
end