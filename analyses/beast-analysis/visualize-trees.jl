using gcdyn, JLD2, Plots, Random, StatsBase

function main()
	# This is computed in a separate script
	discretization_table = Dict([1.1973728522068754, 3.1370222155629772] => 1.4959713153479637, [-0.0002917189005149, 0.09106099779661217] => 0.0, [0.5796933611135875, 1.1973728522068754] => 0.8642148598083795, [-5.7429821755606145, -0.0002917189005149] => -0.4568874976254108, [0.09106099779661217, 0.5796933611135875] => 0.35395573620012577)
	
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

			plot(
				tree;
				colorscheme=:diverging_bkr_55_10_c35_n256,
				midpoint=0,
				reverse_colorscheme=true,
				title="$gc_name STATE_$i",
				dpi=500,
				size=(1000, 700),
				legendtitle="Affinity bin"
			)
			png("$directory_name/tree-STATE_$i.png")
		end
	end
end

main()