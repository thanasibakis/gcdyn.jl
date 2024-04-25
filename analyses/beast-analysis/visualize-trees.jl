using gcdyn, JLD2, Plots, Random, StatsBase

function main()
	# This is computed in a separate script
	discretization_table = Dict([0.7118957155228802, 1.414021615333377] => 1.0158145231080074, [1.414021615333377, 4.510046694952103] => 2.1957206408364116, [0.09499050830860184, 0.7118957155228802] => 0.4104891647333908, [-0.07325129835947444, 0.09499050830860184] => 0.0, [-8.4036117593617, -0.07325129835947444] => -1.224171991580991)

	mkpath("out/tree-visualizations/")
	
	# Get one from each GC
	for germinal_center_dir in readdir("data/jld2-with-affinities/"; join=true)
		tree::TreeNode{Float64} = load_object(joinpath(germinal_center_dir, "tree-STATE_50000000.jld2"))

		map_types!(tree) do affinity
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

		plot(tree; title="$(basename(germinal_center_dir)) STATE_50000000", dpi=500, size=(750, 500), legendtitle="Affinity bin")
		png("out/tree-visualizations/$(basename(germinal_center_dir))-STATE_50000000.png")
	end

	# # Get many from one GC
	Random.seed!(0)
	germinal_center_dir = readdir("data/jld2-with-affinities/"; join=true) |> sample

	for i in (5:5:45) * 1000000
		tree::TreeNode{Float64} = load_object(joinpath(germinal_center_dir, "tree-STATE_$i.jld2"))

		map_types!(tree) do affinity
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

		plot(tree; title="$(basename(germinal_center_dir)) STATE_$i", dpi=500, size=(750, 500), legendtitle="Affinity bin")
		png("out/tree-visualizations/$(basename(germinal_center_dir))-STATE_$i.png")
	end
end

main()