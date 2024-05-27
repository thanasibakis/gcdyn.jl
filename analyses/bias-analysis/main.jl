using gcdyn, CSV, DataFrames, Turing

function main()
    truth = ConstantBranchingProcess(1.8, 1, 0, 1, 0, [1])
    chns = run_simulations(truth, 2)
    
    dfs = map(chns |> collect) do (num_trees, chns_dict)
        DataFrame(
            num_trees = num_trees,
            corrected_λ = map(chns_dict[:corrected]) do chn
                median(chn[:λ])
            end,
            original_λ = map(chns_dict[:original]) do chn
                median(chn[:λ])
            end,
            corrected_μ = map(chns_dict[:corrected]) do chn
                median(chn[:μ])
            end,
            original_μ = map(chns_dict[:original]) do chn
                median(chn[:μ])
            end
        )
    end

    CSV.write("posterior-medians.csv", vcat(dfs...))
end

@model function ConditionedModel(trees, present_time)
    λ ~ LogNormal(1.5, 1)
    μ ~ LogNormal(0, 0.5)

    sampled_model = ConstantBranchingProcess(λ, μ, 0, 1, 0, [1])

    Turing.@addlogprob! sum(gcdyn.stadler_appx_loglikelihood(sampled_model, tree, present_time) for tree in trees)
end

@model function OriginalModel(trees, present_time)
    λ ~ LogNormal(1.5, 1)
    μ ~ LogNormal(0, 0.5)

    sampled_model = ConstantBranchingProcess(λ, μ, 0, 1, 0, [1])

    Turing.@addlogprob! sum(gcdyn.stadler_appx_unconditioned_loglikelihood(sampled_model, tree, present_time) for tree in trees)
end

function run_simulations(truth, present_time)
    NUM_TREESETS = 100

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
        20 => Dict(
            :corrected => Vector{Chains}(undef, NUM_TREESETS),
            :original => Vector{Chains}(undef, NUM_TREESETS)
        ),
        50 => Dict(
            :corrected => Vector{Chains}(undef, NUM_TREESETS),
            :original => Vector{Chains}(undef, NUM_TREESETS)
        )
    )

    for (num_trees, chns_dict) in chns
        Threads.@threads for i in 1:NUM_TREESETS
            trees = rand_tree(truth, present_time, truth.type_space[1], num_trees)

            chns_dict[:corrected][i] = sample(
                ConditionedModel(trees, present_time),
                Gibbs(
                    MH(:λ => x -> LogNormal(log(x), 0.2)),
                    MH(:μ => x -> LogNormal(log(x), 0.2))
                ),
                2000
            )

            chns_dict[:original][i] = sample(
                OriginalModel(trees, present_time),
                Gibbs(
                    MH(:λ => x -> LogNormal(log(x), 0.2)),
                    MH(:μ => x -> LogNormal(log(x), 0.2))
                ),
                2000
            )
        end
    end

    chns
end

main()