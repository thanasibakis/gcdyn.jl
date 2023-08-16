using gcdyn, DataFrames, Turing, StatsPlots;

truth = MultitypeBranchingProcess(2, 1, 0, 1:3, 1, 0, 2);

@model function FullModel(trees::Vector{TreeNode})
    λ ~ LogNormal(1.5, 1)
    μ ~ LogNormal(0, 0.3)

    sampled_model = MultitypeBranchingProcess(
        λ, μ, truth.γ, truth.state_space, truth.transition_matrix, truth.ρ, truth.σ, truth.present_time
    )

    Turing.@addlogprob! loglikelihood(sampled_model, trees)
end;


@model function AppxModel(trees::Vector{TreeNode})
    λ ~ LogNormal(1.5, 1)
    μ ~ LogNormal(0, 0.3)

    sampled_model = MultitypeBranchingProcess(
        λ, μ, truth.γ, truth.state_space, truth.transition_matrix, truth.ρ, truth.σ, truth.present_time
    )

    Turing.@addlogprob! sum(gcdyn.stadler_appx_loglikelhood(sampled_model, tree) for tree in trees)
end;

trees = rand_tree(truth, 50, truth.state_space[1]);

chns_appx = sample(
    AppxModel(trees),
    MH(
        :λ => x -> LogNormal(log(x), 0.2),
        :μ => x -> LogNormal(log(x), 0.3)
    ),
    1000
)

chns_full = sample(
    FullModel(trees),
    MH(
        :λ => x -> LogNormal(log(x), 0.2),
        :μ => x -> LogNormal(log(x), 0.3)
    ),
    1000
)

plot(chns)