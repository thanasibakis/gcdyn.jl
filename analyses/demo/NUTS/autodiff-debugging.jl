using gcdyn, ForwardDiff, OrdinaryDiffEq

function p(μ)
	Γ = [-1 0.5 0.25 0.25; 2 -4 1 1; 2 2 -5 1; 0.125 0.125 0.25 -0.5]
	model = VaryingTypeChangeRateBranchingProcess(1, 5, 1.5, 1, μ, 1, Γ, 1, 0, [2, 4, 6, 8], 3)

    solve(
        ODEProblem{true}(
            gcdyn.dp_dt!,
            fill(1 - model.ρ, size(model.type_space)),
            (0, model.present_time),
            model
        ),
        Tsit5();
        isoutofdomain = (p, args, t) -> any(x -> x < 0 || x > 1, p),
        save_everystep = false,
        save_start = false,
        reltol = 1e-3,
        abstol = 1e-3
    ).u[end][1]
end

ForwardDiff.derivative.(p, 0.05:0.5:5)