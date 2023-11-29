using CSV, gcdyn, CategoricalArrays, DataFrames, Distributions, JLD2, JSON, StatsPlots, Turing

germinal_centers = Vector{Vector{TreeNode}}()

for file in readdir("output/jld2/")
	trees = load_object(joinpath("output/jld2/", file))
	push!(germinal_centers, trees)
end

function bin_states!(trees; visualize=true)
	all_nodes = [node for tree in trees for node in PostOrderTraversal(tree)]

    # The bins are evenly spaced quantiles for the affinities in (-2, inf), including the min and max
	states = [node.state for node in all_nodes]
	num_bins = 5
	cutoffs = quantile(
		filter(x -> x >= -2, states),
		collect(0:(1/(num_bins-1)):1)
	)
	pushfirst!(cutoffs, minimum(states))

	binned_states = convert.(
        Float64,
        cut(states, cutoffs; extend=true, labels=(from, to, i; leftclosed, rightclosed) -> to)
    )
	
	for (node, binned_state) in zip(all_nodes, binned_states)
		node.state = binned_state
	end

    if visualize
        ENV["GKSwstype"] = "100" # operate plotting in headless mode
	    histogram(states; normalize=:pdf, fill="grey", alpha=0.2)
	    histogram!(binned_states; normalize=:pdf)
	    png("binned-states.png")
    end
end

all_trees = (tree for gc in germinal_centers for tree in gc)
bin_states!(all_trees)

state_space = unique(node.state for tree in all_trees for node in PostOrderTraversal(tree)) |> sort
present_time = maximum(leaf.t for tree in all_trees for leaf in LeafTraversal(tree))

birth_waiting_times = Dict(st=>[] for st in state_space)

for tree in all_trees
	for node in PostOrderTraversal(tree)
		if node.event == :birth
			push!(birth_waiting_times[node.state], node.t - node.up.t)
		end
	end
end

histogram(birth_waiting_times; normalize=:pdf)

for state in state_space
	waiting_times = birth_waiting_times[state]

	# Print a summary
	println("State: $state")
	println("Mean: $(mean(waiting_times))")
	println("MLE of rate: $(1/mean(waiting_times))")
	println("")
end

transition_matrix = zeros(length(state_space), length(state_space))
for tree in all_trees
	for node in PostOrderTraversal(tree)
		if node.event == :mutation
			j = findfirst(state_space .== node.state)
			i = findfirst(state_space .== node.up.state)
			transition_matrix[i, j] += 1
		end
	end
end
transition_matrix ./= sum(transition_matrix, dims=2)

m = SigmoidalBirthRateBranchingProcess(1, 0, 0.5, 0.5, 0.8, 0.8, state_space, transition_matrix, 1, 0, present_time)