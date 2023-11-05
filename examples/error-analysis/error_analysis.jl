using gcdyn, Turing, StatsPlots

@model function CorrectedModel(trees::Vector{TreeNode}, truth::AbstractBranchingProcess)
    λ ~ LogNormal(1.5, 1)
    μ ~ LogNormal(0, 0.5)

    sampled_model = ConstantRateBranchingProcess(λ, μ, truth.γ, truth.state_space, truth.ρ, truth.σ, truth.present_time)

    Turing.@addlogprob! sum(gcdyn.stadler_appx_loglikelhood(sampled_model, tree) for tree in trees)
end

@model function OriginalModel(trees::Vector{TreeNode}, truth::AbstractBranchingProcess)
    λ ~ LogNormal(1.5, 1)
    μ ~ LogNormal(0, 0.5)

    sampled_model = ConstantRateBranchingProcess(λ, μ, truth.γ, truth.state_space, truth.ρ, truth.σ, truth.present_time)

    Turing.@addlogprob! sum(gcdyn.stadler_appx_unconditioned_loglikelhood(sampled_model, tree) for tree in trees)
end

function run_simulations(truth)
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
            trees = rand_tree(truth, num_trees, truth.state_space[1])
    
            chns_dict[:corrected][i] = sample(
                CorrectedModel(trees, truth),
                MH(
                    :λ => x -> LogNormal(log(x), 0.2),
                    :μ => x -> LogNormal(log(x), 0.2)
                ),
                2000
            )
    
            chns_dict[:original][i] = sample(
                OriginalModel(trees, truth),
                MH(
                    :λ => x -> LogNormal(log(x), 0.2),
                    :μ => x -> LogNormal(log(x), 0.2)
                ),
                2000
            )
        end
    end

    chns
end

function visualize_original(chns, truth)
    λ_plts = []
    μ_plts = []
    
    for n in sort(collect(keys(chns)))
        chns_dict = chns[n]

        # Birth rate plots
        medians_original = map(chns_dict[:original]) do chn
            median(get(chn, :λ)[1])
        end
    
        plt = histogram(medians_original; alpha=0.7, label="Original", normalize=:pdf, fill="grey")
        vline!([truth.λ]; label="Truth", linewidth=4, color="orange")
        xlims!(0, 3)
        title!("$n trees", titlefontsize=10)
    
        push!(λ_plts, plt)
    
        # Death rate plots
        medians_original = map(chns_dict[:original]) do chn
            median(get(chn, :μ)[1])
        end
    
        plt = histogram(medians_original; alpha=0.7, label="Original", normalize=:pdf, fill="grey")
        vline!([truth.μ]; label="Truth", linewidth=4, color="orange")
        xlims!(0, 3)
        title!("$n trees", titlefontsize=10)
    
        push!(μ_plts, plt)
    end
    
    plot(λ_plts...; layout=(1, 4), thickness_scaling=0.75, dpi=300, size=(1000, 250), plot_title="Birth rate sampling distributions", plot_titlefontsize=12)
    xlabel!("Posterior median")
    ylabel!("Density")
    png("birth-rate-errors-original-only.png")
    
    plot(μ_plts...; layout=(1, 4), thickness_scaling=0.75, dpi=300, size=(1000, 250), plot_title="Death rate sampling distributions", plot_titlefontsize=12)
    xlabel!("Posterior median")
    ylabel!("Density")
    png("death-rate-errors-original-only.png")
end

function visualize_both(chns, truth)
    λ_plts = []
    μ_plts = []
    
    for n in sort(collect(keys(chns)))
        chns_dict = chns[n]

        # Birth rate plots
        medians_corrected = map(chns_dict[:corrected]) do chn
            median(get(chn, :λ)[1])
        end
        medians_original = map(chns_dict[:original]) do chn
            median(get(chn, :λ)[1])
        end
    
        plt = histogram(medians_original; alpha=0.7, label="Original", normalize=:pdf, fill="grey")
        histogram!(medians_corrected; alpha=0.7, label="Corrected", normalize=:pdf, fill="lightblue")
        vline!([truth.λ]; label="Truth", linewidth=4, color="orange")
        xlims!(0, 3)
        title!("$n trees", titlefontsize=10)
    
        push!(λ_plts, plt)
    
        # Death rate plots
        medians_corrected = map(chns_dict[:corrected]) do chn
            median(get(chn, :μ)[1])
        end
        medians_original = map(chns_dict[:original]) do chn
            median(get(chn, :μ)[1])
        end
    
        plt = histogram(medians_original; alpha=0.7, label="Original", normalize=:pdf, fill="grey")
        histogram!(medians_corrected; alpha=0.7, label="Corrected", normalize=:pdf, fill="lightblue")
        vline!([truth.μ]; label="Truth", linewidth=4, color="orange")
        xlims!(0, 3)
        title!("$n trees", titlefontsize=10)
    
        push!(μ_plts, plt)
    end
    
    plot(λ_plts...; layout=(1, 4), thickness_scaling=0.75, dpi=300, size=(1000, 250), plot_title="Birth rate sampling distributions", plot_titlefontsize=12)
    xlabel!("Posterior median")
    ylabel!("Density")
    png("birth-rate-errors.png")
    
    plot(μ_plts...; layout=(1, 4), thickness_scaling=0.75, dpi=300, size=(1000, 250), plot_title="Death rate sampling distributions", plot_titlefontsize=12)
    xlabel!("Posterior median")
    ylabel!("Density")
    png("death-rate-errors.png")
end

function main()
    truth = ConstantRateBranchingProcess(1.8, 1, 0, 1:2, 1, 0, 2)
    chns = run_simulations(truth)
    visualize_original(chns, truth)
    visualize_both(chns, truth)
end

main()