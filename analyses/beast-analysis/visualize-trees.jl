using gcdyn, JLD2, Plots, Random, StatsBase

function main()
	# This is computed in a separate script
	discretization_table = Dict([0.7118957155228802, 1.414021615333377] => 1.0158145231080074, [1.414021615333377, 4.510046694952103] => 2.1957206408364116, [0.09499050830860184, 0.7118957155228802] => 0.4104891647333908, [-0.07325129835947444, 0.09499050830860184] => 0.0, [-8.4036117593617, -0.07325129835947444] => -1.224171991580991)

	mkpath("out/tree-visualizations") # Make this before we multithread

	Threads.@threads for germinal_center_dir in readdir("data/jld2-with-affinities/"; join=true)
		gc_name = basename(germinal_center_dir)
		directory_name = "out/tree-visualizations/$gc_name"
		mkpath(directory_name)

		for i in (5:5:45) * 1000000
			tree::TreeNode{Float64} = load_object(joinpath(germinal_center_dir, "tree-STATE_$i.jld2"))

			# Don't prune self loops when binning here. We want to visualize when the nucleotide-level mutations occurred too
			map_types!(tree; prune_self_loops=false) do affinity
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

			plot(tree; title="$gc_name STATE_$i", dpi=500, size=(1000, 500), legendtitle="Affinity bin")
			png("$directory_name/tree-STATE_$i.png")
		end
	end
end

main()