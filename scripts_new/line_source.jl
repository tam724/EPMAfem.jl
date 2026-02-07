using Revise
using EPMAfem
using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using EPMAfem.Gridap
using LinearAlgebra
using Plots
using EPMAfem.HCubature
using LaTeXStrings
using EPMAfem.Distributions
using EPMAfem.HCubature

struct PlaneSourceEquations{S} <: EPMAfem.AbstractPNEquations end
EPMAfem.number_of_elements(::PlaneSourceEquations) = 1
EPMAfem.number_of_scatterings(::PlaneSourceEquations) = 1
EPMAfem.stopping_power(::PlaneSourceEquations, e, ϵ) = 1.0
EPMAfem.absorption_coefficient(eq::PlaneSourceEquations, e, ϵ) = 0.0 # 1.0 
EPMAfem.scattering_coefficient(eq::PlaneSourceEquations, e, i, ϵ) = 0.0 #1.0
function EPMAfem.mass_concentrations(::PlaneSourceEquations, e, x)
    return 1.0 # x[1] < 0.5 ? 1.0 : 2.0
end
EPMAfem.scattering_kernel(::PlaneSourceEquations{Inf}, e, i) = μ -> 1/(4π)
@generated μ₀(::PlaneSourceEquations{T}) where T = return :( $ (2π*hquadrature(μ -> exp(-T*(μ-1)^2), -1, 1)[1]))
EPMAfem.scattering_kernel(eq::PlaneSourceEquations{T}, e, i) where T = μ -> exp(-T*(μ-1)^2)/μ₀(eq)

energy_model = 0:0.01:1.0

T = Inf
N = 47
equations = PlaneSourceEquations{T}()
space_model1 = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.5, 1.5), (100)); plus=(name=lagrangian, order=1, conformity=:H1), minus=(name=lagrangian, order=0, conformity=:L2))
space_model2 = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.5, 1.5), (50)); plus=(name=lagrangian, order=2, conformity=:H1), minus=(name=lagrangian, order=1, conformity=:L2))
space_model3 = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.5, 1.5), (33)); plus=(name=lagrangian, order=3, conformity=:H1), minus=(name=lagrangian, order=2, conformity=:L2))
space_model4 = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.5, 1.5), (25)); plus=(name=lagrangian, order=4, conformity=:H1), minus=(name=lagrangian, order=3, conformity=:L2))

direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(N, 1, :OE)
function make_sol(space_model)
    model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)
    problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())

    @show N, EPMAfem.n_basis(problem)

    # source / boundary condition (here: zero)
    source = EPMAfem.Rank1DiscretePNVector(false, model, EPMAfem.cpu(), zeros(EPMAfem.n_basis(model).nϵ), (p=zeros(EPMAfem.n_basis(model).nx.p), m=zeros(EPMAfem.n_basis(model).nx.m)), (p=zeros(EPMAfem.n_basis(model).nΩ.p), m=zeros(EPMAfem.n_basis(model).nΩ.m)))

    # initial condition
    Mp = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.plus(EPMAfem.space_model(model)), EPMAfem.SpaceModels.plus(EPMAfem.space_model(model)))
    Mm = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.minus(EPMAfem.space_model(model)), EPMAfem.SpaceModels.minus(EPMAfem.space_model(model)))

    # system = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.cg, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));

    # function solver(A::PNLazyMatrices.BlockMatrix)
    #     _A = PNLazyMatrices.A(A)
    #     _B = PNLazyMatrices.B(A)
    #     _C = PNLazyMatrices.D(A)
    #     _A_inv = lazy(Krylov.cg, _A)
    #     _C_inv = PNLazyMatrices.cache(LinearAlgebra.inv!(-_C))
    #     return lazy(Krylov.tricg, _A_inv, _B, _C_inv)
    # end

    # system = EPMAfem.implicit_midpoint2(problem, solver);
    # system = EPMAfem.implicit_midpoint2(problem, Krylov.minres);

    # x3
    system = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.cg(M=cache(inv(diagonal(PNLazyMatrices.blocks(A)[1])))), cache ∘ inv));
    # system = EPMAfem.implicit_midpoint2(problem, Krylov.minres);


    # σ = 0.03
    σ = 0.08
    init_x(x) = 1/(σ*sqrt(2π))*exp(-1/2*((x[1]-0.0)^2)/σ^2) #  + (x[2]-0.0)^2  
    init_Ω(Ω) = Distributions.pdf(VonMisesFisher([1, 0, 0], 20.0), [Ω...])
    bxp = Mp \ EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫R_μv(init_x), EPMAfem.space_model(model), EPMAfem.SpaceModels.plus(EPMAfem.space_model(model)))
    bxm = Mm \ EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫R_μv(init_x), EPMAfem.space_model(model), EPMAfem.SpaceModels.minus(EPMAfem.space_model(model)))
    bΩp = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.plus(EPMAfem.direction_model(model)))
    bΩm = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.minus(EPMAfem.direction_model(model)))

    hΩp = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(Ω -> 1.0), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.plus(EPMAfem.direction_model(model)))
    hΩm = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(Ω -> 1.0), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.minus(EPMAfem.direction_model(model)))

    initial_condition = EPMAfem.allocate_solution_vector(system)
    ψ0p, ψ0m = EPMAfem.pmview(initial_condition, model)
    copy!(ψ0p, bxp .* bΩp')
    copy!(ψ0m, bxm .* bΩm')

    sol =  EPMAfem.IterableDiscretePNSolution(system, source, initial_solution=initial_condition, step_callback=(ϵ, _) -> @show ϵ);
    return (sol=sol, hΩ=(p=hΩp, m=hΩm), model=model)
end

zeroth = make_sol(space_model1);
first = make_sol(space_model2);
second = make_sol(space_model3);
third = make_sol(space_model4);


ws = third.sol.system.BM⁻¹.ws
A = unlazy(zeroth.sol.system.BM⁻¹.A.args[1].args[1])

M = zeros(30300, 30300)
e = zeros(30300)
for i in 1:30300
    e[i] = 1.0
    mul!(@view(M[:, i]), A, e)
    e[i] = 0.0
end

M
spy(M)
A*rand(30300)

30300@gif for (ϵ, ψ) in second.sol
    ψp, ψm = EPMAfem.pmview(ψ, second.model)
    f = EPMAfem.SpaceModels.interpolable((p=collect(ψp)*second.hΩ.p, m=collect(ψm)*second.hΩ.m), EPMAfem.space_model(second.model))
    plot(-1.5:0.01:1.5, x -> f(VectorValue(x)))
    # heatmap(-1.5:0.005:1.5, -1.5:0.005:1.5, (x, y) -> f(VectorValue(x, y)), aspect_ratio=:equal)
end

@gif for ((_, ψ0), (_, ψ1), (_, ψ2), (_, ψ3)) in zip(zeroth.sol, first.sol, second.sol, third.sol)
    ψp0, ψm0 = EPMAfem.pmview(ψ0, zeroth.model)
    ψp1, ψm1 = EPMAfem.pmview(ψ1, first.model)
    ψp2, ψm2 = EPMAfem.pmview(ψ2, second.model)
    ψp3, ψm3 = EPMAfem.pmview(ψ3, third.model)

    f0 = EPMAfem.SpaceModels.interpolable((p=collect(ψp0)*zeroth.hΩ.p, m=collect(ψm0)*zeroth.hΩ.m), EPMAfem.space_model(zeroth.model))
    f1 = EPMAfem.SpaceModels.interpolable((p=collect(ψp1)*first.hΩ.p, m=collect(ψm1)*first.hΩ.m), EPMAfem.space_model(first.model))
    f2 = EPMAfem.SpaceModels.interpolable((p=collect(ψp2)*second.hΩ.p, m=collect(ψm2)*second.hΩ.m), EPMAfem.space_model(second.model))
    f3 = EPMAfem.SpaceModels.interpolable((p=collect(ψp3)*third.hΩ.p, m=collect(ψm3)*third.hΩ.m), EPMAfem.space_model(third.model))
    plot(-0.5:0.01:1.5, x -> f0(VectorValue(x)), label="order 0/1")
    plot!(-0.5:0.01:1.5, x -> f1(VectorValue(x)), label="order 1/2")
    plot!(-0.5:0.01:1.5, x -> f2(VectorValue(x)), label="order 2/3")
    plot!(-0.5:0.01:1.5, x -> f3(VectorValue(x)), label="order 3/4")
    ylims!(-0.3, 6)
    # heatmap(-1.5:0.005:1.5, -1.5:0.005:1.5, (x, y) -> f(VectorValue(x, y)), aspect_ratio=:equal)
end


(ϵ, ψ), next = iterate(sol)
ψp, ψm = EPMAfem.pmview(ψ, model)
f = EPMAfem.SpaceModels.interpolable((p=collect(ψp)*bΩp, m=collect(ψm)*bΩm), EPMAfem.space_model(model))
plot(-1.5:0.0001:1.5, x -> f(VectorValue(x)))

(ϵ1, ψ1), next1 = iterate(sol, next)
@profview (ϵ1, ψ1), next1 = iterate(sol, next)
ψp1, ψm1 = EPMAfem.pmview(ψ1, model)
f1 = EPMAfem.SpaceModels.interpolable((p=collect(ψp1)*bΩp, m=collect(ψm1)*bΩm), EPMAfem.space_model(model))
plot!(-1.5:0.0001:1.5, x -> f1(VectorValue(x)))


probe = EPMAfem.PNProbe(model, EPMAfem.cpu(), ϵ=0.0)

ψp, ψm = probe(sol)
@profview ψp, ψm = probe(sol)
f = EPMAfem.SpaceModels.interpolable((p=collect(ψp)*bΩp, m=collect(ψm)*bΩm), EPMAfem.space_model(model))
heatmap(-1.5:0.005:1.5, -1.5:0.005:1.5, (x, y) -> f(VectorValue(x, y)))
plot(-1.5:0.0001:1.5, x -> f(VectorValue(x, 0.0)))
plot(-1.5:0.0001:1.5, x -> f(VectorValue(x)))


p1 = plot(-1.5:0.01:1.5, x -> f(VectorValue(x)))
p1 = plot!(p1, -1.5:0.01:1.5, x -> f(VectorValue(x)))
title!(p1, "solution at end time")


ψ_norm = Float64[]
ψ_mass = Float64[]

Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω->1/(4π)) |> EPMAfem.architecture(problem)
xp, xm = EPMAfem.SpaceModels.eval_basis(EPMAfem.space_model(model), x -> 1.0) |> EPMAfem.architecture(problem)

for (ϵ, ψ) in sol
    ψp, ψm = EPMAfem.pmview(ψ, model)
    push!(ψ_norm, LinearAlgebra.dot(ψp, Mp*ψp) + LinearAlgebra.dot(ψm, Mm*ψm))
    push!(ψ_mass, LinearAlgebra.dot(xp, ψp*Ωp) + LinearAlgebra.dot(xm, ψm*Ωm))
    # f = EPMAfem.SpaceModels.interpolable((p=collect(ψp)*bΩp, m=collect(ψm)*bΩm), EPMAfem.space_model(model))
    # plot(-1.5:0.01:1.5, x -> f(VectorValue(x)))
    # heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> f(VectorValue(x, y)))
    # ylims!(-0.1, 15)
end

p2 = plot(ψ_norm .- ψ_norm[1])
p2 = plot!(p2, ψ_norm .- ψ_norm[1])
# xlims!(p2, 0, 50)
# ylims!(p2, -0.00000001, 0.00000001)
title!(p2, "energy")

p3 = plot(ψ_mass .- ψ_mass[1])
p3 = plot!(p3, ψ_mass .- ψ_mass[1])
# xlims!(p3, 0, 50)
# ylims!(p3, -0.0001, 0.0001)
title!(p3, "mass")



plot(p1, p2, p3, size=(1000, 600), legend=nothing)



A = Matrix(system.BM.A.args[1])
B = Matrix(system.BM.A.args[2])
C = -Diagonal(Matrix(system.BM.A.args[4]))

M = [A B
-B' C]

A_eig = eigen(A)
C_eig = eigen(C)
M_eig = eigen(M)

scatter(M_eig.values)
# lump the whole matrix (?)
M_lump = Diagonal(sum(M; dims=2)[:])
PM_var = M_lump\M

PM_var_eig = eigen(PM_var)
plot(scatter(M_eig.values),
    scatter(PM_var_eig.values))

# lump the whole matrix (?)
M_diag = Diagonal(diag(M))
PM_var2 = M_diag\M

PM_var2_eig = eigen(PM_var2)

plot(scatter(M_eig.values),
    scatter(PM_var_eig.values),
    scatter(PM_var2_eig.values))

# preconditioning with diagonal blocks
PM = [I(size(A, 1)) A\B
-C\B' I(size(C, 1))]

PM_eig = eigen(PM)

scatter(PM_eig.values)

# preconditioning with diagonal approximation of A
Ã = Diagonal(A)
PM2 = [Ã\A Ã\B
-C\B' I(size(C, 1))]

PM2_eig = eigen(PM2)

scatter!(PM2_eig.values)

# preconditioning with lump approximation of A
A_lump = Diagonal(sum(A; dims=1)[:])
PM2_var = [A_lump\A A_lump\B
-C\B' I(size(C, 1))]

PM2_eig_var = eigen(PM2_var)

plot(scatter(PM_eig.values),
    scatter(PM2_eig.values),
    scatter(PM2_eig_var.values))

# preconditioning with schur complement over A
S_A = C + transpose(B) * (A\B)
PM3 = [I(size(A, 1)) A\B
-S_A\(B') S_A \ C]

PM3_eig = eigen(PM3)

scatter!(PM3_eig.values)

# preconditioning with approximate schur complement over A
S_Ã = C + Diagonal(transpose(B) * (Ã \ B))
PM4 = [Ã\A Ã\B
-S_Ã\(B') S_Ã \ C]

PM4_eig = eigen(PM4)

# preconditioning with approximate schur complement over A
PM4_var = [A\A A\B
-S_Ã\(B') S_Ã \ C]

PM4_eig_var = eigen(PM4_var)

# preconditioning with schur complement over C
S_C = A + B * (C \ transpose(B))
PM5 = [S_C\A S_C\B
-C\(B') I(size(C, 1))]

PM5_eig = eigen(PM5)

# preconditioning with approximate schur complement over C
S_C̃ = Diagonal(A + B * (C \ transpose(B)))
PM6 = [S_C̃\A S_C̃\B
-C\(B') I(size(C, 1))]

PM6_eig = eigen(PM6)

α = 0.1
plot(scatter(M_eig.values, alpha=α, aspect_ratio=:equal),
    scatter(PM_eig.values, alpha=α, aspect_ratio=:equal),
    scatter(PM2_eig.values, alpha=α, aspect_ratio=:equal),
    scatter(PM3_eig.values, alpha=α, aspect_ratio=:equal),
    scatter(PM4_eig.values, alpha=α, aspect_ratio=:equal),
    scatter(PM4_eig_var.values, alpha=α, aspect_ratio=:equal),
    scatter(PM5_eig.values, alpha=α, aspect_ratio=:equal),
    scatter(PM6_eig.values, alpha=α, aspect_ratio=:equal))
