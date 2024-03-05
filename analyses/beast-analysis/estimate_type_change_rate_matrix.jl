using CategoricalArrays, CSV, DataFrames, Distributions, LinearAlgebra

function compute_discretization_table(affinities; state_space_size=5)		
	# Create bins from evenly spaced quantiles, then discretize states to the medians of their bins
	cutoffs = quantile(affinities, 0:(1/state_space_size):1)
	bin_table = DataFrame(
		state=affinities,
		bin=cut(affinities, cutoffs; extend=true)
	)
	bin_table = DataFrames.transform(groupby(bin_table, :bin), :state => median => :binned_state)
	bin_table = select(bin_table, Not(:state)) |> unique
	
	parse_interval(row) = parse.(Float64, split(convert(String, row.bin)[2:end-1], ", "))
	
	Dict(parse_interval(row) => row.binned_state for row in eachrow(bin_table))
end

function get_discretization(affinity, discretization_table)
	for (bin, value) in discretization_table
		if bin[1] <= affinity < bin[2]
			return value
		end
	end

	if all(bin[2] <= affinity for bin in keys(discretization_table))
		return maximum(values(discretization_table))
	elseif all(affinity < bin[1] for bin in keys(discretization_table))
		return minimum(values(discretization_table))
	else
		error("Affinity $affinity not in any bin!")
	end
end

function compute_rate_matrix(starting_sequences, state_space)
	type_change_history = let
		command = pipeline(`bin/simulate-s5f-mutations`; stdin=IOBuffer(join(starting_sequences, "\n")))
		output = read(command, String) |> strip
		rows = split(output, "\n")
		mutation_histories = split.(rstrip.(rows, ';'), ";")

		map(mutation_histories) do mutations
			mutations = [parse.(Float64, split(mutation, ':')) for mutation in mutations]
			[(get_discretization(from_affinity, discretization_table), duration, get_discretization(to_affinity, discretization_table)) for (from_affinity, duration, to_affinity) in mutations]
		end
	end

	rate_matrix = zeros(length(state_space), length(state_space))
	duration_times = zeros(length(state_space))

	for history in type_change_history
		for (from_state, duration, to_state) in history
			i = findfirst(==(from_state), state_space)
			j = findfirst(==(to_state), state_space)
			
			if i != j
				rate_matrix[i, j] += 1
			end

			duration_times[i] += duration
		end
	end

	rate_matrix ./= duration_times
	rate_matrix[diagind(rate_matrix)] = -sum.(eachrow(rate_matrix))

	rate_matrix
end
