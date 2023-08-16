using gcdyn, DataFrames, Turing, StatsPlots

expit(x) = 1 / (1 + exp(-x));
sigmoid(x, xscale, xshift, yscale, yshift) = yscale * expit(xscale * (x - xshift)) + yshift;

λ_truth = x -> sigmoid(x, 1, 5, 3, 1);

truth = MultitypeBranchingProcess(λ_truth, 1, 2, 1:3, 1, 0, 2);

@model function Model(trees::Vector{TreeNode})
    xscale ~ Gamma(2, 1)
    xshift ~ Normal(5, 1)
    yscale ~ Gamma(2, 1)
    yshift ~ Gamma(1, 1)

    λ = x -> sigmoid(x, xscale, xshift, yscale, yshift)
    μ ~ LogNormal(0, 0.3)

    Turing.@addlogprob! loglikelihood(
       MultitypeBranchingProcess(λ, μ, truth.γ, truth.state_space, truth.transition_matrix, truth.ρ, truth.σ, truth.present_time),
       trees
    )
end;

chns = Vector{Chains}(undef, 2);

Threads.@threads for i in 1:2
    trees = rand_tree(truth, 20, truth.state_space[1])
    chns[i] = sample(
        Model(trees),
        MH(
            :xscale => x -> LogNormal(log(x), 0.2),
            :xshift => x -> Normal(x, 1),
            :yscale => x -> LogNormal(log(x), 0.2),
            :yshift => x -> LogNormal(log(x), 0.2),
            :μ => x -> LogNormal(log(x), 0.1)
        ),
        1000
    )
end;

medians = map(chns) do chn
    map(median, get_params(chn))
end |> DataFrame

select!(medians, Not(:lp))
