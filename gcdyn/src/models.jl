abstract type AbstractTreeModel end

struct NaiveModel <: AbstractTreeModel
    λ::Real
    μ::Real
    ρ::Real
    σ::Real
    present_time::Real

    function NaiveModel(λ, μ, ρ, σ, present_time)
        if λ < 0 || μ < 0 
            throw(ArgumentError("Rate parameters must be positive"))
        elseif ρ < 0 || ρ > 1
            throw(ArgumentError("ρ must be between 0 and 1"))
        elseif present_time < 0
            throw(ArgumentError("Time must be positive"))
        end

        if σ != 1
            throw(ArgumentError("Naive model assumes σ = 1"))
        end

        return new(λ, μ, ρ, σ, present_time)
    end
end

struct StadlerAppxModel <: AbstractTreeModel
    λ::Real
    μ::Real
    ρ::Real
    σ::Real
    present_time::Real

    function StadlerAppxModel(λ, μ, ρ, σ, present_time)
        if λ < 0 || μ < 0 
            throw(ArgumentError("Rate parameters must be positive"))
        elseif ρ < 0 || ρ > 1
            throw(ArgumentError("ρ must be between 0 and 1"))
        elseif σ < 0 || σ > 1
            throw(ArgumentError("σ must be between 0 and 1"))
        elseif present_time < 0
            throw(ArgumentError("Time must be positive"))
        end

        return new(λ, μ, ρ, σ, present_time)
    end
end

struct StadlerAppxModelOriginal <: AbstractTreeModel
    λ::Real
    μ::Real
    ρ::Real
    σ::Real
    present_time::Real

    function StadlerAppxModelOriginal(λ, μ, ρ, σ, present_time)
        if λ < 0 || μ < 0 
            throw(ArgumentError("Rate parameters must be positive"))
        elseif ρ < 0 || ρ > 1
            throw(ArgumentError("ρ must be between 0 and 1"))
        elseif σ < 0 || σ > 1
            throw(ArgumentError("σ must be between 0 and 1"))
        elseif present_time < 0
            throw(ArgumentError("Time must be positive"))
        end

        return new(λ, μ, ρ, σ, present_time)
    end
end

function Distributions.loglikelihood(model::AbstractTreeModel, trees::Vector{TreeNode})
    return sum(logpdf(model, tree) for tree in trees)
end

function Distributions.logpdf(model::NaiveModel, tree::TreeNode)
    result = 0
    λ, μ, ρ = model.λ, model.μ, model.ρ

    for node in AbstractTrees.PostOrderDFS(tree.children[1])
        if node.event == :birth
            result += logpdf(Exponential(1 / (λ + μ)), node.t - node.up.t) + log(λ / (λ + μ))
        elseif node.event == :sampled_death
            result += logpdf(Exponential(1 / (λ + μ)), node.t - node.up.t) + log(μ / (λ + μ))
        elseif node.event == :unsampled_death
            throw(ArgumentError("Tree contains unsampled death event"))
        elseif node.event == :sampled_survival
            result += log(1 - cdf(Exponential(1 / (λ + μ)), node.t - node.up.t)) + log(ρ)
        elseif node.event == :unsampled_survival
            result += log(1 - cdf(Exponential(1 / (λ + μ)), node.t - node.up.t)) + log(1 - ρ)
        end
    end

    return result
end

function Distributions.logpdf(model::StadlerAppxModel, tree::TreeNode)
    result = 0
    λ, μ, ρ, σ, present_time = model.λ, model.μ, model.ρ, model.σ, model.present_time

    for node in AbstractTrees.PostOrderDFS(tree.children[1])
        Λ = λ + μ
        c = √(Λ^2 - 4 * μ * (1 - σ) * λ)
        x = (-Λ - c) / 2
        y = (-Λ + c) / 2

        helper(t) = (y + λ * (1 - ρ)) * exp(-c * t) - x - λ * (1 - ρ)

        t_s = present_time - node.up.t
        t_e = present_time - node.t

        log_f_N = c * (t_e - t_s) + 2 * (log(helper(t_e)) - log(helper(t_s)))

        result += log_f_N

        if node.event == :birth
            result += log(λ)
        elseif node.event == :sampled_death
            result += log(σ) + log(μ)
        elseif node.event == :sampled_survival
            result += log(ρ)
        else
            throw(ArgumentError("Tree contains incompatible event $(node.event)"))
        end
    end

    # Condition on tree not getting rejected as a stub
    # Let p be the extinction (or more generally, emptiness) prob
    Λ = λ + μ
    root_time = present_time - 0
    c = √(Λ^2 - 4 * μ * (1 - σ) * λ)
    x = (-Λ - c) / 2
    y = (-Λ + c) / 2

    p = (
        -1
        / λ
        * ((y + λ * (1 - ρ)) * x * exp(-c * root_time) - y * (x + λ * (1 - ρ)))
        / ((y + λ * (1 - ρ)) * exp(-c * root_time) - (x + λ * (1 - ρ)))
    )

    # p = 1 - (λ - μ) / (λ - μ * exp(-(λ - μ) * present_time))

    result -= log(1 - p)

    return result
end

function Distributions.logpdf(model::StadlerAppxModelOriginal, tree::TreeNode)
    result = 0
    λ, μ, ρ, σ, present_time = model.λ, model.μ, model.ρ, model.σ, model.present_time

    for node in AbstractTrees.PostOrderDFS(tree.children[1])
        Λ = λ + μ
        c = √(Λ^2 - 4 * μ * (1 - σ) * λ)
        x = (-Λ - c) / 2
        y = (-Λ + c) / 2

        helper(t) = (y + λ * (1 - ρ)) * exp(-c * t) - x - λ * (1 - ρ)

        t_s = present_time - node.up.t
        t_e = present_time - node.t

        log_f_N = c * (t_e - t_s) + 2 * (log(helper(t_e)) - log(helper(t_s)))

        result += log_f_N

        if node.event == :birth
            result += log(λ)
        elseif node.event == :sampled_death
            result += log(σ) + log(μ)
        elseif node.event == :sampled_survival
            result += log(ρ)
        else
            throw(ArgumentError("Tree contains incompatible event $(node.event)"))
        end
    end

    return result
end

function rand_tree(model::AbstractTreeModel, n::Int; reject_stubs::Bool=true)
    trees = Vector{TreeNode}(undef, n)

    i = 1
    while i <= n
        trees[i] = TreeNode(:root, 0, 0)
        evolve!(trees[i], model.present_time, model.λ, model.μ, model.ρ, model.σ)
        prune!(trees[i])

        if reject_stubs && length(trees[i].children) == 0
            continue
        end

        i += 1
    end

    return trees
end
