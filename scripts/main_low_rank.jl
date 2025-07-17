using Revise
using EPMAfem

using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using EPMAfem.Gridap
using LinearAlgebra
include("plot_overloads.jl")

# space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0, -1, 1), (50, 50)))
# direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(11, 2)

space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0), 100))
direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(21, 1)

equations = EPMAfem.PNEquations()
model = EPMAfem.DiscretePNModel(space_model, 0:0.01:1.0, direction_model)

problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())

# system = EPMAfem.implicit_midpoint(problem, EPMAfem.PNKrylovMinresSolver)
# systemm1 = EPMAfem.implicit_midpoint(problem, EPMAfem.PNSchurSolver)

# system2 = EPMAfem.implicit_midpoint2(problem, Krylov.minres)
# system3 = EPMAfem.implicit_midpoint2(problem, Krylov.gmres)
# system4 = EPMAfem.implicit_midpoint2(problem, \)

system4 = EPMAfem.implicit_midpoint2(problem, \)
system5 = EPMAfem.implicit_midpoint2(problem, (PNLazyMatrices.schur_complement, Krylov.minres))
system6 = EPMAfem.implicit_midpoint_dlr(problem; max_rank=5);

excitation = EPMAfem.pn_excitation([(x=0.0, y=0.0)], [0.7], [VectorValue(-1.0, 0.0, 0.0)])
extraction = EPMAfem.PNExtraction([0.1, 0.2], equations)

discrete_rhs = EPMAfem.discretize_rhs(excitation, model, EPMAfem.cpu())[1]
# discrete_extr = EPMAfem.discretize_extraction(extraction, model, EPMAfem.cpu())[1].vector

x = range(-1, 1, length=101)
f(x) = exp(-100*x^2)
plot(f.(x))

discrete_rhs2 = EPMAfem.Rank1DiscretePNVector(false, model, EPMAfem.cpu(), discrete_rhs.bϵ, f.(x), zeros(size(discrete_rhs.bΩp)))
discrete_rhs2.bΩp[1] = -1.0

# sol4 = system4 * discrete_rhs2
sol5 = system5 * discrete_rhs2
sol6 = system6 * discrete_rhs2

anim = @animate for ((i5, ψ5), (i6, ψ6)) in zip(sol5, sol6)
    @show i5, i6
    # ψp4, ψm4 = EPMAfem.pmview(ψ4, model)
    # func4 = EPMAfem.SpaceModels.interpolable(ψp4[:, 1] |> collect, EPMAfem.space_model(model))
    # plot(-1.0:0.01:0, func4.interp, swapxy=true, label="backslash")

    ψp5, ψm5 = EPMAfem.pmview(ψ5, model)
    func5 = EPMAfem.SpaceModels.interpolable(ψp5[:, 1] |> collect, EPMAfem.space_model(model))
    plot(-1.0:0.01:0, func5.interp, swapxy=true, label="schur minres")

    ψp6, ψm6 = EPMAfem.pmview(ψ6, model)
    func6 = EPMAfem.SpaceModels.interpolable(ψp6[:, 1] |> collect, EPMAfem.space_model(model))
    plot!(-1.0:0.01:0, func6.interp, swapxy=true, label="dlr")

    U, S, Vt = EPMAfem.USVt(ψ6)
    plot!(size=(1000, 1000))
end
gif(anim)

sol = system * discrete_rhs;
solm1 = systemm1 * discrete_rhs;
sol2 = system2 * discrete_rhs;
sol3 = system3 * discrete_rhs;
sol4 = system4 * discrete_rhs;
sol5 = system5 * discrete_rhs;

@time discrete_extr * (system * discrete_rhs)
@profview discrete_extr * (systemm1 * discrete_rhs)
@btime discrete_extr * (systemm1 * discrete_rhs)
@time discrete_extr * (system2 * discrete_rhs)
@time discrete_extr * (system3 * discrete_rhs)
@time discrete_extr * (system4 * discrete_rhs)
@profview discrete_extr * (system5 * discrete_rhs)
@btime discrete_extr * (system5 * discrete_rhs)

@gif for ((i1, ψ1), (i2, ψ2), (i3, ψ3), (i4, ψ4), (i5, ψ5)) in zip(sol, sol2, sol3, sol4, sol5)
    ψp1, ψm1 = EPMAfem.pmview(ψ1, model)
    func1 = EPMAfem.SpaceModels.interpolable(ψp1[:, 1] |> collect, EPMAfem.space_model(model))
    # p1 = heatmap(-1.0:0.01:0, -1:0.01:1, func1.interp, swapxy=true, aspect_ratio=:equal)
    p1 = plot(-1.0:0.01:0, func1.interp, swapxy=true, aspect_ratio=:equal)

    ψp2, ψm2 = EPMAfem.pmview(ψ2, model)
    func2 = EPMAfem.SpaceModels.interpolable(ψp2[:, 1] |> collect, EPMAfem.space_model(model))
    p2 = plot(-1.0:0.01:0, func2.interp, swapxy=true, aspect_ratio=:equal)

    ψp3, ψm3 = EPMAfem.pmview(ψ3, model)
    func3 = EPMAfem.SpaceModels.interpolable(ψp3[:, 1] |> collect, EPMAfem.space_model(model))
    p3 = plot(-1.0:0.01:0, func3.interp, swapxy=true, aspect_ratio=:equal)

    ψp4, ψm4 = EPMAfem.pmview(ψ4, model)
    func4 = EPMAfem.SpaceModels.interpolable(ψp4[:, 1] |> collect, EPMAfem.space_model(model))
    p4 = plot(-1.0:0.01:0, func4.interp, swapxy=true, aspect_ratio=:equal)

    ψp5, ψm5 = EPMAfem.pmview(ψ5, model)
    func5 = EPMAfem.SpaceModels.interpolable(ψp5[:, 1] |> collect, EPMAfem.space_model(model))
    p5 = plot(-1.0:0.01:0, func5.interp, swapxy=true, aspect_ratio=:equal)
    plot(p1, p2, p3, p4, p5)
end

## test the lowranksolution
(_, (nxp, nxm), (nΩp, nΩm)) = EPMAfem.n_basis(problem)

Up, Sp, Vtp = svd(rand(nxp, nΩp))
tUp, tSp, tVtp = Up[:, 1:5], Diagonal(Sp[1:5]), Vtp[1:5, :]
tXp = tUp*tSp*tVtp
Xm = rand(nxm, nΩm)

sol.rank[] = 5

solU, solS, solVt = EPMAfem.USVt(sol)
solU .= tUp
solS .= tSp
solVt .= tVtp

sol._xm .= @view(Xm[:])

rec_x = rand(nxp*nΩp + nxm*nΩm)
EPMAfem.vec!(rec_x, sol)

@view(rec_x[1:nxp*nΩp]) ≈ @view(tXp[:])
@view(rec_x[nxp*nΩp+1:end]) ≈ @view(Xm[:])
### end test


BM = unlazy(BMl)
system = EPMAfem.implicit_midpoint(problem, EPMAfem.PNKrylovMinresSolver)
system.A.sym[] = false
nd, ne, nσ = EPMAfem.n_sums(problem)

for i in 1:ne
    system.coeffs.a[i] = rand()
    coeffs.a[i][] = system.coeffs.a[i]
    for j in 1:nσ
        system.coeffs.c[i][j] = rand()
        coeffs.c[i][j][] = system.coeffs.c[i][j]
    end
end

x = rand(size(BM, 2))
y1 = rand(size(BM, 1))
y2 = rand(size(BM, 1))

@time mul!(y1, system.A, x);
@time mul!(y2, BM, x);

y1 ≈ y2
