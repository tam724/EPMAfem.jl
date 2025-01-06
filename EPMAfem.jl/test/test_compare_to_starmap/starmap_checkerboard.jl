
## CHECKERBOARD
@concrete struct CheckerboardEquations <: EPMAfem.AbstractPNEquations end
# 1: purely absorbing material
# 2: scattering material
EPMAfem.number_of_elements(::CheckerboardEquations) = 2 
EPMAfem.number_of_scatterings(::CheckerboardEquations) = 1
EPMAfem.stopping_power(::CheckerboardEquations, e, ϵ) = 1.0
EPMAfem.absorption_coefficient(::CheckerboardEquations, e, ϵ) = e == 1 ? 10.0 : 1.0
EPMAfem.scattering_coefficient(::CheckerboardEquations, e, i, ϵ) = e == 1 ? 0.0 : 1.0
EPMAfem.electron_scattering_kernel(eq::CheckerboardEquations, e, i, μ) = 1.0/4π
lattice((x, y)) = (ceil(Int, x), ceil(Int, y))
function EPMAfem.mass_concentrations(::CheckerboardEquations, e, (x, y))
    if lattice((x, y)) in [   (2, 6),                         (6, 6),
                            (3, 5),         (5, 5),
                    (2, 4),                         (6, 4),
                            (3, 3),         (5, 3),
                    (2, 2),         (4, 2),         (6, 2)]
        # absorbing material
        return e == 1 ? 1.0 : 0.0
    else
        # scattering material
        return e == 2 ? 1.0 : 0.0
    end
end
source_space_distribution(::CheckerboardEquations, (x, y)) = lattice((x, y)) == (4, 4) ? -1.0 : 0.0

function discretize_excitation(eq::CheckerboardEquations, discrete_model::EPMAfem.PNGridapModel, arch::EPMAfem.PNArchitecture)
    T = EPMAfem.base_type(arch)

    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels

    space_mdl = EPMAfem.space_model(discrete_model)
    direction_mdl = EPMAfem.direction_model(discrete_model)

    μϵ = Vector{T}([1.0 for ϵ ∈ EPMAfem.energy_model(discrete_model)])
    μxp = SM.assemble_linear(SM.∫R_μv(x -> source_space_distribution(eq, x)), space_mdl, SM.even(space_mdl)) |> arch
    μΩp = SH.assemble_linear(SH.∫S²_hv(Ω -> 1.0), direction_mdl, SH.even(direction_mdl)) |> arch

    return EPMAfem.Rank1DiscretePNVector(false, discrete_model, arch, μϵ, μxp, μΩp)
end

function compute(N)
    eq = CheckerboardEquations()
    nx, ny = 250, 250
    space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((0, 7, 0, 7), (nx, ny)))
    direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(N, 2)
    
    model = EPMAfem.PNGridapModel(space_model, 0.0:0.2:3.2, direction_model)
    discrete_problem = EPMAfem.discretize_problem(eq, model, EPMAfem.cuda())
    discrete_system = EPMAfem.schurimplicitmidpointsystem(discrete_problem)

    rhs = discretize_excitation(eq, model, EPMAfem.cuda())

    solution = discrete_system*rhs
    discrete_sol = nothing
    for (idx, sol) in solution @show idx; discrete_sol = sol end
    return reshape(EPMAfem.pview(discrete_sol, model)[:, 1] |> collect, (nx+1, ny+1))
end

@time ψ3 = compute(3)
@time ψ5 = compute(5)
@time ψ15 = compute(15)
@time ψ27 = compute(27)

fix0(x, ϵ=1e-10) = x < ϵ ? ϵ : x

p1 = heatmap(log.(fix0.(ψ3))', aspect_ratio=:equal, clim=(-7, 0), color=:jet)
title!("N=3")
p2 = heatmap(log.(fix0.(ψ5))', aspect_ratio=:equal, clim=(-7, 0), color=:jet)
title!("N=5")
p3 = heatmap(log.(fix0.(ψ15))', aspect_ratio=:equal, clim=(-7, 0), color=:jet)
title!("N=15")
p4 = heatmap(log.(fix0.(ψ27))', aspect_ratio=:equal, clim=(-7, 0), color=:jet)
title!("N=27")
plot(p1, p2, p3, p4, size=(1500, 1500))

# @gif for (i, ϵ) in solution
#     @show ϵ
#     ψ = EPMAfem.current_solution(discrete_system)
#     ψ0 = reshape(EPMAfem.pview(ψ, model)[:, 1] |> collect, (101, 101))
#     heatmap(ψ0, aspect_ratio=:equal)
# end

# ψ0 = reshape(EPMAfem.pview(EPMAfem.current_solution(discrete_system), model)[:, 1] |> collect, (101, 101))
# heatmap(log.(fix0.(ψ0))', aspect_ratio=:equal, clim=(-7, 0), color=:jet)