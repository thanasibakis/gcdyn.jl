println("Loading packages...")
using CSV, gcdyn, DataFrames, Distributions, JLD2, JSON, LinearAlgebra, StatsPlots, Turing

function main()
	println("Reading trees...")
    germinal_centers = Vector{Vector{TreeNode}}()

    #for file in readdir("output/jld2/")
	for file in readdir("output/jld2/")[1:2]
        trees = load_object(joinpath("output/jld2/", file))
        push!(germinal_centers, trees)
    end

	println("Determining state bins...")
	bin_mapping = get_bin_mapping()
	state_space = values(bin_mapping) |> collect |> sort

	function get_discretization(affinity)
		for (bin, value) in bin_mapping
			if bin[1] <= affinity < bin[2]
				return value
			end
		end

		if all(bin[2] <= affinity for bin in keys(bin_mapping))
			return maximum(values(bin_mapping))
		elseif all(affinity < bin[1] for bin in keys(bin_mapping))
			return minimum(values(bin_mapping))
		else
			error("Affinity $affinity not in any bin!")
		end
	end
	
	println("Estimating transition matrix...")
	treeset = collect(sample(gc) for gc in germinal_centers)
	sequences = [node.info[:sequence] for tree in treeset for node in PostOrderTraversal(tree)]
	mutated_affinities = pipeline(`bin/simulate-s5f-mutations`; stdin=IOBuffer(join(sequences, "\n"))) |>
		(command -> read(command, String)) |>
		strip |>
		(text -> split(text, "\n")) |>
		(rows -> split.(rstrip.(rows, ';'), ";"))

	transition_matrix = zeros(length(state_space), length(state_space))
	duration_times = zeros(length(state_space))

	for mutations in mutated_affinities
		for mutation in mutations
			from_affinity, duration, to_affinity = parse.(Float64, split(mutation, ':'))

			i = findfirst(state_space .== get_discretization(from_affinity))
			j = findfirst(state_space .== get_discretization(to_affinity))
			
			if i != j
				transition_matrix[i, j] += 1
			end

			duration_times[i] += duration
		end
	end

	transition_matrix ./= duration_times
	transition_matrix[diagind(transition_matrix)] = -sum.(eachrow(transition_matrix))
	
	println("Estimated transition matrix:")
	println(transition_matrix, "\n")

	println("Sampling from posterior...")
	present_time = maximum([node.t for tree in treeset for node in PostOrderTraversal(tree)])
	model = Model(treeset, state_space, transition_matrix, present_time)
	posterior_samples = sample(
		model,
		Gibbs(
			# MH(:xscale => x -> LogNormal(log(x), 0.2)),
			# MH(:xshift => x -> Normal(x, 0.7)),
			# MH(:yscale => x -> LogNormal(log(x), 0.2)),
			# MH(:yshift => x -> LogNormal(log(x), 0.2)),
			MH(:λ => x -> LogNormal(log(x), 0.2)),
			MH(:μ => x -> LogNormal(log(x), 0.2)),
			MH(:γ => x -> LogNormal(log(x), 0.2))
		),
		1 # 5000
	) |> DataFrame

	println("Done!")
end

function get_bin_mapping()	
	STATE_SPACE_SIZE = 5

	# We will use the 10x sequences to determine the state bins
	affinities = CSV.read("../../lib/gcreplay/analysis/output/10x/data.csv", DataFrame).delta_bind_CGG
	
	# Create bins from evenly spaced quantiles, then discretize states to the medians of their bins
	cutoffs = quantile(affinities, 0:(1/STATE_SPACE_SIZE):1)
	bin_table = DataFrame(
		state=affinities,
		bin=cut(affinities, cutoffs; extend=true)
	)
	bin_table = transform(groupby(bin_table, :bin), :state => median => :binned_state)
	bin_table = select(bin_table, Not(:state)) |> unique
	
	parse_interval(row) = parse.(Float64, split(convert(String, row.bin)[2:end-1], ", "))
	
	Dict(parse_interval(row) => row.binned_state for row in eachrow(bin_table))
end

@model function Model(trees, state_space, transition_matrix, present_time)
    # xscale  ~ Gamma(2, 1)
    # xshift  ~ Normal(0, 1)
    # yscale  ~ Gamma(1, 1)
    # yshift  ~ Gamma(1, 0.5)
	λ		~ LogNormal(0, 1)
    μ       ~ LogNormal(0, 1)
    γ       ~ LogNormal(0, 1)

    # Only compute loglikelihood if we're not sampling from the prior
    # if DynamicPPL.getsampler(__context__) != DynamicPPL.SampleFromPrior()
	if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
		# sampled_model = SigmoidalBirthRateBranchingProcess(
		# 	xscale, xshift, yscale, yshift, μ, γ, state_space, transition_matrix, 1, 0, present_time
		# )
		sampled_model = ConstantRateBranchingProcess(λ, μ, γ, state_space, transition_matrix, 1, 0, present_time)

        Turing.@addlogprob! loglikelihood(sampled_model, trees; reltol=1e-3, abstol=1e-3)
    end
end

# julia> transition_matrix
# 5×5 Matrix{Float64}:
#  0.0       0.75817    0.0686275  0.163399   0.00980392
#  0.734854  0.0        0.213471   0.0381326  0.0135424
#  0.582357  0.350093   0.0        0.0441086  0.0234423
#  0.570198  0.176713   0.195432   0.0        0.0576563
#  0.414557  0.0779272  0.125791   0.381725   0.0

# julia> mle = optimize(model, MLE(), Optim.Options(iterations=10_000, allow_f_increases=true))
# ┌ Warning: Optimization did not converge! You may need to correct your model or adjust the Optim parameters.
# └ @ TuringOptimExt ~/.julia/packages/Turing/UCuzt/ext/TuringOptimExt.jl:241
# ModeResult with maximized lp of -Inf
# [0.8213822990414615, 0.5510818348258846, 3.6581944935508055]

# julia> mle = optimize(model, MLE(), Optim.Options(iterations=10_000, allow_f_increases=true))
# ┌ Warning: Optimization did not converge! You may need to correct your model or adjust the Optim parameters.
# └ @ TuringOptimExt ~/.julia/packages/Turing/UCuzt/ext/TuringOptimExt.jl:241
# ModeResult with maximized lp of -Inf
# [1.1175977920799343, 1.418080181668995, 0.7143853228312714]

# julia> mle = optimize(model, MLE(), Optim.Options(iterations=10_000, allow_f_increases=true))
# ┌ Warning: Optimization did not converge! You may need to correct your model or adjust the Optim parameters.
# └ @ TuringOptimExt ~/.julia/packages/Turing/UCuzt/ext/TuringOptimExt.jl:241
# ModeResult with maximized lp of -Inf
# [0.6045182824674427, 4.802343391692815, 1.1565297809861894]