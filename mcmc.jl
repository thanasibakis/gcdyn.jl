using gcdyn, Turing, StatsPlots

truth = StadlerAppxModel(1.8, 1, 1, 0, 2);

@model function SingleBirthRateModel(trees::Vector{TreeNode})
    λ ~ LogNormal(1.5, 1)
    μ ~ LogNormal(0, 0.3)
    Turing.@addlogprob! loglikelihood(
        StadlerAppxModel(λ, μ, truth.ρ, truth.σ, truth.present_time),
        trees
    )
end;

chns = Vector{Chains}(undef, 200);

Threads.@threads for i in 1:200
    trees = rand_tree(truth, 20)
    chns[i] = sample(
        SingleBirthRateModel(trees),
        MH(
            :λ => x -> LogNormal(log(x), 0.1),
            :μ => x -> LogNormal(log(x), 0.1)
        ),
        1000
    )
end;

map(chns) do chn
    median(get(chn, :λ)[1])
end |> histogram

vline!([truth.λ]; label = "Truth", linewidth = 4)

map(chns) do chn
    median(get(chn, :μ)[1])
end |> histogram

vline!([truth.μ]; label = "Truth", linewidth = 4)
