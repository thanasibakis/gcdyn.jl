using gcdyn, Turing, StatsPlots

truth = StadlerAppxModel(1.8, 1, 1, 0, 2);

@model function CorrectedModel(trees::Vector{TreeNode})
    λ ~ LogNormal(1.5, 1)
    μ ~ LogNormal(0, 0.3)
    Turing.@addlogprob! loglikelihood(
        StadlerAppxModel(λ, μ, truth.ρ, truth.σ, truth.present_time),
        trees
    )
end;

@model function OriginalModel(trees::Vector{TreeNode})
    λ ~ LogNormal(1.5, 1)
    μ ~ LogNormal(0, 0.3)
    Turing.@addlogprob! loglikelihood(
        StadlerAppxModelOriginal(λ, μ, truth.ρ, truth.σ, truth.present_time),
        trees
    )
end;

NUM_TREESETS = 400;

# num_trees => Dict(:corrected => [Chains], :original => [Chains])
chns = Dict(
    1 => Dict(
        :corrected => Vector{Chains}(undef, NUM_TREESETS),
        :original => Vector{Chains}(undef, NUM_TREESETS)
    ),
    5 => Dict(
        :corrected => Vector{Chains}(undef, NUM_TREESETS),
        :original => Vector{Chains}(undef, NUM_TREESETS)
    ),
    10 => Dict(
        :corrected => Vector{Chains}(undef, NUM_TREESETS),
        :original => Vector{Chains}(undef, NUM_TREESETS)
    ),
    20 => Dict(
        :corrected => Vector{Chains}(undef, NUM_TREESETS),
        :original => Vector{Chains}(undef, NUM_TREESETS)
    )
);

sample_model(model) = sample(
    model,
    MH(
        :λ => x -> LogNormal(log(x), 0.1),
        :μ => x -> LogNormal(log(x), 0.1)
    ),
    2000
);

for (num_trees, chns_dict) in chns
    Threads.@threads for i in 1:NUM_TREESETS
        trees = rand_tree(truth, num_trees)
        chns_dict[:corrected][i] = sample_model(CorrectedModel(trees))
        chns_dict[:original][i] = sample_model(OriginalModel(trees))
    end
end;

λ_plts = []
μ_plts = []

for (n, chns_dict) in chns
    # Birth rate plots
    medians_corrected = map(chns_dict[:corrected]) do chn
        median(get(chn, :λ)[1])
    end
    medians_original = map(chns_dict[:original]) do chn
        median(get(chn, :λ)[1])
    end

    plt = histogram(medians_original; alpha=0.7, label="Original", normalize=:pdf)
    histogram!(medians_corrected; alpha=0.7, label="Corrected", normalize=:pdf)
    vline!([truth.λ]; label="Truth", linewidth=4)
    title!("$n trees")

    push!(λ_plts, plt)

    # Death rate plots
    medians_corrected = map(chns_dict[:corrected]) do chn
        median(get(chn, :μ)[1])
    end
    medians_original = map(chns_dict[:original]) do chn
        median(get(chn, :μ)[1])
    end

    plt = histogram(medians_original; alpha=0.7, label="Original", normalize=:pdf)
    histogram!(medians_corrected; alpha=0.7, label="Corrected", normalize=:pdf)
    vline!([truth.μ]; label="Truth", linewidth=4)
    title!("$n trees")

    push!(μ_plts, plt)
end;

plot(λ_plts...; layout=(2, 2), plot_title = "Birth rate sampling distributions", thickness_scaling=0.75, dpi=300)
xlabel!("Posterior median")
ylabel!("Density")
png("birth_rate_errors.png")

plot(μ_plts...; layout=(2, 2), plot_title = "Death rate sampling distributions", thickness_scaling=0.75, dpi=300)
xlabel!("Posterior median")
ylabel!("Density")
png("death_rate_errors.png")
