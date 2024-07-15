using Gridap
using SparseArrays

## Gridap Helper function

function assemble_bilinear(a, args, U, V)
    u = get_trial_fe_basis(U)
    v = get_fe_basis(V)
    matcontribs = a(u, v, args...)
    data = Gridap.FESpaces.collect_cell_matrix(U, V, matcontribs)
    return assemble_matrix(SparseMatrixAssembler(U, V), data)
end

function assemble_linear(b, args, U, V)
    v = get_fe_basis(V)
    veccontribs = b(v, args...)
    data = Gridap.FESpaces.collect_cell_vector(V, veccontribs)
    return assemble_vector(SparseMatrixAssembler(U, V), data)
end

## general
function number_of_basis_functions(solver)
    x = (p=num_free_dofs(solver.U[1]), m=num_free_dofs(solver.U[2]))
    Ω = (p=length([m for m in SphericalHarmonicsMatrices.get_moments(solver.PN, nd(solver)) if SphericalHarmonicsMatrices.is_even(m...)]), m=length([m for m in SphericalHarmonicsMatrices.get_moments(solver.PN, nd(solver)) if SphericalHarmonicsMatrices.is_odd(m...)]))
    return (x=x, Ω=Ω)
end

## solver functions
function build_solver(model, PN, n_elem)
    V = MultiFieldFESpace([TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1), TestFESpace(model, ReferenceFE(lagrangian, Float64, 0), conformity=:L2)])
    U = MultiFieldFESpace([TrialFESpace(V[1]), TrialFESpace(V[2])])

    R = Triangulation(model)
    dx = Measure(R, 2)
    ∂R = BoundaryTriangulation(model)
    dΓ = Measure(∂R, 2)
    n = get_normal_vector(∂R)

    return (U=U, V=V, model=(model=model, R=R, dx=dx, ∂R=∂R, dΓ=dΓ, n=n), PN=PN, n_elem=n_elem)
end

number_of_dimensions(::DiscreteModel{N}) where N = N
function nd(::DiscreteModel{N}) where N
    return Val{N}()
end

function nd(solver)
    return nd(solver.model.model)
end

function material_space(model)
    return FESpace(model, ReferenceFE(lagrangian, Float64, 0), conformity=:L2)
end

# defs of bilinear forms
∂z(u, ::DiscreteModel{1}) = dot(VectorValue(1.0), ∇(u))

∂z(u, ::DiscreteModel{2}) = dot(VectorValue(1.0, 0.0), ∇(u))
∂x(u, ::DiscreteModel{2}) = dot(VectorValue(0.0, 1.0), ∇(u))

∂z(u, ::DiscreteModel{3}) = dot(VectorValue(1.0, 0.0, 0.0), ∇(u))
∂x(u, ::DiscreteModel{3}) = dot(VectorValue(0.0, 1.0, 0.0), ∇(u))
∂y(u, ::DiscreteModel{3}) = dot(VectorValue(0.0, 0.0, 1.0), ∇(u))

∫ρuv(u, v, ρ, (model, R, dx, ∂R, dΓ, n)) = ∫(ρ*u*v)dx
∫uv(u, v, (model, R, dx, ∂R, dΓ, n)) = ∫(u*v)dx
∫v(v, (model, R, dx, ∂R, dΓ, n)) = ∫(v)dx

∫∂zu_v(u, v, (model, R, dx, ∂R, dΓ, n)) = ∫(∂z(u, model)*v)dx
∫∂xu_v(u, v, (model, R, dx, ∂R, dΓ, n)) = ∫(∂x(u, model)*v)dx
∫∂yu_v(u, v, (model, R, dx, ∂R, dΓ, n)) = ∫(∂y(u, model)*v)dx

∫u_∂zv(u, v, (model, R, dx, ∂R, dΓ, n)) = ∫(u*∂z(v, model))dx
∫u_∂xv(u, v, (model, R, dx, ∂R, dΓ, n)) = ∫(u*∂x(v, model))dx
∫u_∂yv(u, v, (model, R, dx, ∂R, dΓ, n)) = ∫(u*∂y(v, model))dx

∫∂u_v(::DiscreteModel{1}) = (∫∂zu_v, )
∫∂u_v(::DiscreteModel{2}) = (∫∂zu_v, ∫∂xu_v)
∫∂u_v(::DiscreteModel{3}) = (∫∂zu_v, ∫∂xu_v, ∫∂yu_v)

∫u_∂v(::DiscreteModel{1}) = (∫u_∂zv, )
∫u_∂v(::DiscreteModel{2}) = (∫u_∂zv, ∫u_∂xv)
∫u_∂v(::DiscreteModel{3}) = (∫u_∂zv, ∫u_∂xv, ∫u_∂yv)

nz(n, ::DiscreteModel{1}) = dot(n, VectorValue(1.0))

nz(n, ::DiscreteModel{2}) = dot(n, VectorValue(1.0, 0.0))
nx(n, ::DiscreteModel{2}) = dot(n, VectorValue(0.0, 1.0))

nz(n, ::DiscreteModel{3}) = dot(n, VectorValue(1.0, 0.0, 0.0))
nx(n, ::DiscreteModel{3}) = dot(n, VectorValue(0.0, 1.0, 0.0))
ny(n, ::DiscreteModel{3}) = dot(n, VectorValue(0.0, 0.0, 1.0))

∫absnz_uv(u, v, (model, R, dx, ∂R, dΓ, n)) = ∫(abs(nz(n, model))*u*v)dΓ
∫absnx_uv(u, v, (model, R, dx, ∂R, dΓ, n)) = ∫(abs(nx(n, model))*u*v)dΓ
∫absny_uv(u, v, (model, R, dx, ∂R, dΓ, n)) = ∫(abs(ny(n, model))*u*v)dΓ

∫absn_uv(::DiscreteModel{1}) = (∫absnz_uv, )
∫absn_uv(::DiscreteModel{2}) = (∫absnz_uv, ∫absnx_uv)
∫absn_uv(::DiscreteModel{3}) = (∫absnz_uv, ∫absnx_uv, ∫absny_uv)

## space assembly
function update_space_matrices!(X, solver, mass_concentrations)
    U = solver.U
    V = solver.V
    model = solver.model
    X.Xpp .= [assemble_bilinear(∫ρuv, (ρi, model), U[1], V[1]) for ρi ∈ mass_concentrations]
    X.Xmm .= [assemble_bilinear(∫ρuv, (ρi, model), U[2], V[2]) for ρi ∈ mass_concentrations]
end

function update_extractions!(μhs, solver, mass_concentrations)
    for (i, μx) ∈ enumerate(μhs.x)
        μsp, μsm = assemble_space_source(solver, mass_concentrations[i])
        μhs.x[i].p .= μsp
        μhs.x[i].m .= μsm 
    end
    # μh_x = [assemble_space_source(solver, x -> μx(x, mass_concentrations[i])) for (i, μx) ∈ enumerate(μ.x)]
end

function assemble_space_matrices(solver, mass_concentrations)
    U = solver.U
    V = solver.V
    model = solver.model
    Xpp = [assemble_bilinear(∫ρuv, (ρi, model), U[1], V[1]) for ρi ∈ mass_concentrations]
    Xmm = [assemble_bilinear(∫ρuv, (ρi, model), U[2], V[2]) for ρi ∈ mass_concentrations]
    dXpm = [assemble_bilinear(a, (model, ), U[1], V[2]) for a ∈ ∫∂u_v(model.model)]
    dXmp = [assemble_bilinear(a, (model, ), U[2], V[1]) for a ∈ ∫u_∂v(model.model)]
    ∂Xpp = [assemble_bilinear(a, (model, ), U[1], V[1]) for a ∈ ∫absn_uv(model.model)]
    return (Xpp=Xpp, Xmm=Xmm, dXpm=dXpm, dXmp=dXmp, ∂Xpp=∂Xpp)
end

∫μv(v, μ, (model, R, dx, ∂R, dΓ, n)) = ∫(μ*v)dx
∫ngv(v, g, (model, R, dx, ∂R, dΓ, n)) = ∫(nz(n, model)*g*v)dΓ

function assemble_space_source((U, V, model), μ_x)
    return (p=assemble_linear(∫μv, (μ_x, model), U[1], V[1]), m=assemble_linear(∫μv, (μ_x, model), U[2], V[2]))
end

function assemble_space_boundary((U, V, model), g_x)
    # since we only reqire the even moments on the boundary, the second here is 0
    # return assemble_linear(∫ngv, (g_x, model), U[1], V[1]), assemble_linear(∫ngv, (g_x, model), U[2], V[2])
    return (p=round.(sparse(assemble_linear(∫ngv, (g_x, model), U[1], V[1])), digits=8), m=spzeros(num_free_dofs(U[2])))
end

## direction assembly
# function get_transport_matrices(N, nd::Val{1})
#     Apm1 = assemble_transport_matrix(N, Val{3}(), :pm, nd)
#     Amp1 = assemble_transport_matrix(N, Val{3}(), :mp, nd)

#     return (Apm1, ), (Amp1, )
# end

# function get_transport_matrices(N, nd::Val{2})
#     Apm1 = assemble_transport_matrix(N, Val{3}(), :pm, nd)
#     Amp1 = assemble_transport_matrix(N, Val{3}(), :mp, nd)

#     Apm2 = assemble_transport_matrix(N, Val{1}(), :pm, nd)
#     Amp2 = assemble_transport_matrix(N, Val{1}(), :mp, nd)
#     return (Apm1, Apm2), (Amp1, Amp2)
# end

# function get_boundary_matrices(N, ::Val{1})
#     ∂App1 = assemble_boundary_matrix(N, Val(3), :pp, nd)
#     return (round.(∂App1, digits=8), )
# end

# function get_boundary_matrices(N, ::Val{2})
#     ∂App1 = assemble_boundary_matrix(N, Val(3), :pp, nd)
#     ∂App2 = assemble_boundary_matrix(N, Val(1), :pp, nd)
#     return (round.(∂App1, digits=8), round.(∂App2, digits=8))
# end

value(::Val{N}) where N = N

space_directions(::DiscreteModel{1}) = (Val(3), )
space_directions(::DiscreteModel{2}) = (Val(3), Val(1))
space_directions(::DiscreteModel{3}) = (Val(3), Val(1), Val(2))


function assemble_direction_matrices(solver, scattering_kernel)
    N = solver.PN
    num_dim = nd(solver)
    dApm = [assemble_transport_matrix(N, dir, :pm, num_dim) for dir ∈ (Val(3), Val(1), Val(2))[1:value(num_dim)]]
    dAmp = [assemble_transport_matrix(N, dir, :mp, num_dim) for dir ∈ (Val(3), Val(1), Val(2))[1:value(num_dim)]]
    ∂App = [round.(assemble_boundary_matrix(N, dir, :pp, num_dim), digits=5) for dir ∈ (Val(3), Val(1), Val(2))[1:value(num_dim)]]
    Kpp, Kmm = assemble_scattering_matrices(N, scattering_kernel, num_dim)
    AIpp = Diagonal(ones(size(Kpp, 1)))
    AImm = Diagonal(ones(size(Kmm, 1)))
    return (dApm=dApm, dAmp=dAmp, ∂App=Matrix.(∂App), Kpp=Kpp, Kmm=Kmm, AIpp=AIpp, AImm=AImm)
end

function semidiscretize_boundary(solver, g)
    N = solver.PN
    num_dim = nd(solver)
    # we assume here that the only boundary with beams is the "upper" z boundary. with outwards boundary normal [0, 0, 1]
    gh_Ω = [assemble_direction_boundary(N, gΩ, [0.0, 0.0, 1.0], num_dim) for gΩ ∈ g.Ω]
    gh_x = [assemble_space_boundary(solver, gx) for gx ∈ g.x]
    return (x=gh_x, Ω=gh_Ω, ϵ=g.ϵ)
end

function semidiscretize_source(solver, μ, mass_concentrations)
    N = solver.PN
    num_dim = nd(solver)
    μh_Ω = [assemble_direction_source(N, μΩ, num_dim) for μΩ ∈ μ.Ω]
    μh_x = [assemble_space_source(solver, μx(mass_concentrations[i])) for (i, μx) ∈ enumerate(μ.x)]
    return (x=μh_x, Ω=μh_Ω, ϵ=μ.ϵ)
end

function projection_matrix(solver)
    projection_matrix = zeros(num_free_dofs(solver.U[1]), num_free_dofs(solver.U[2]))
    for i in 1:num_free_dofs(solver.U[1])
        u = FEFunction(solver.U[1], spzeros(num_free_dofs(solver.U[1])))
        u.free_values[i] = 1.0
        m = interpolate(u, solver.U[2])
        projection_matrix[i, :] .= m.free_values
    end
    return projection_matrix
end

function gram_matrix(solver)
    gram_matrix = assemble_bilinear(∫uv, (solver.model, ), solver.U[2], solver.V[2])
end

import Gridap.TensorValues.⊗
⊗(A, B) = kron(A, B)
⊗ₓ((A, ), (B, )) = kron(A, B)
⊗ₓ((A1, A2), (B1, B2)) = kron(A1, B1) + kron(A2, B2)

using LinearOperators

function Ab_midpoint(ϵ0, ϵ, ψ0, physics, X, Ω, gh)
    Δϵ = ϵ - ϵ0
    @assert Δϵ < 0.0 # should be solved backwards!
    s = physics.s(ϵ)
    s0 = physics.s(ϵ0)
    s_2 = physics.s((ϵ0 + ϵ) / 2.0)
    τ_2 = physics.τ((ϵ0 + ϵ) / 2.0)
    σ_2 = physics.σ((ϵ0 + ϵ) / 2.0)

    ΣsXpp = deepcopy(X.Xpp[1])
    ΣsXmm = deepcopy(X.Xmm[1])
    ΣsXpp.nzval .= (s[1] + s_2[1]).*X.Xpp[1].nzval .+ (s[2] + s_2[2]).*X.Xpp[2].nzval # only binary materials
    ΣsXmm.nzval .= (s[1] + s_2[1]).*X.Xmm[1].nzval .+ (s[2] + s_2[2]).*X.Xmm[2].nzval # only binary materials

    ΣsXpp0 = deepcopy(X.Xpp[1])
    ΣsXmm0 = deepcopy(X.Xmm[1])
    ΣsXpp0.nzval .= (s0[1] + s_2[1]).*X.Xpp[1].nzval .+ (s0[2] + s_2[2]).*X.Xpp[2].nzval # only binary materials
    ΣsXmm0.nzval .= (s0[1] + s_2[1]).*X.Xmm[1].nzval .+ (s0[2] + s_2[2]).*X.Xmm[2].nzval # only binary materials

    # ΣsXpp_2 = s_2[1] .* X.Xpp[1] .+ s_2[2] .* X.Xpp[2]
    # ΣsXmm_2 = s_2[1] .* X.Xmm[1] .+ s_2[2] .* X.Xmm[2]
    
    ΣτXpp_2 = deepcopy(X.Xpp[1])
    ΣτXmm_2 = deepcopy(X.Xmm[1])
    ΣτXpp_2.nzval .= τ_2[1] .* X.Xpp[1].nzval .+ τ_2[2] .* X.Xpp[2].nzval
    ΣτXmm_2.nzval .= τ_2[1] .* X.Xmm[1].nzval .+ τ_2[2] .* X.Xmm[2].nzval
    
    ΣσXpp_2 = deepcopy(X.Xpp[1])
    ΣσXmm_2 = deepcopy(X.Xmm[1])
    ΣσXpp_2.nzval .= σ_2[1] .* X.Xpp[1].nzval .+ σ_2[2] .* X.Xpp[2].nzval
    ΣσXmm_2.nzval .= σ_2[1] .* X.Xmm[1].nzval .+ σ_2[2] .* X.Xmm[2].nzval

    AA_pp = (X.∂Xpp ⊗ₓ Ω.∂App .+ ΣτXpp_2 ⊗ Ω.AIpp .- ΣσXpp_2 ⊗Ω.Kpp)
    AA_mm = (ΣτXmm_2 ⊗Ω.AImm .- ΣσXmm_2 ⊗Ω.Kmm)

    AA_mp = X.dXmp ⊗ₓ Ω.dAmp
    AA_pm = X.dXpm ⊗ₓ Ω.dApm

    A_b1 = LinearOperator(ΣsXpp0 ⊗ Ω.AIpp .- (-Δϵ / 2) .* AA_pp)
    A_b2 = LinearOperator((-Δϵ / 2) .* AA_mp)
    A_b3 = LinearOperator((-(-Δϵ / 2)) .* AA_pm)
    A_b4 = LinearOperator(ΣsXmm0 ⊗Ω.AImm .- (-Δϵ / 2) .* AA_mm)

    A_b = [A_b1 A_b2
        A_b3 A_b4]

    b = A_b*ψ0 - (-Δϵ)*gh((ϵ0 + ϵ) / 2.0)

    A1 = LinearOperator(ΣsXpp ⊗ Ω.AIpp .+ (-Δϵ / 2) .* AA_pp)
    A2 = LinearOperator((-(-Δϵ / 2)) .* AA_mp)
    A3 = LinearOperator((-Δϵ / 2) .* AA_pm)
    A4 = LinearOperator(ΣsXmm ⊗Ω.AImm .+ (-Δϵ / 2) .* AA_mm)

    A = [A1 A2
        A3 A4]
    return A, b
end

function has_same_sparsity_patters(A, B)
    r = sum(abs.(A.rowval .- B.rowval))
    c = sum(abs.(A.colptr .- B.colptr))
    if r + c == 0
        return true
    end
    return false
end


function Abx_midpoint(ϵ0, ϵ, ψ0, physics, X, Ω, μh)
    Δϵ = ϵ - ϵ0
    @assert Δϵ > 0.0 # should be solved forwards!
    s = physics.s(ϵ)
    s0 = physics.s(ϵ0)
    s_2 = physics.s((ϵ0 + ϵ) / 2.0)
    τ_2 = physics.τ((ϵ0 + ϵ) / 2.0)
    σ_2 = physics.σ((ϵ0 + ϵ) / 2.0)

    ΣsXpp = (s[1] + s_2[1]).*X.Xpp[1] .+ (s[2] + s_2[2]).*X.Xpp[2] # only binary materials
    ΣsXmm = (s[1] + s_2[1]).*X.Xmm[1] .+ (s[2] + s_2[2]).*X.Xmm[2] # only binary materials

    ΣsXpp0 = (s0[1] + s_2[1]).*X.Xpp[1] .+ (s0[2] + s_2[2]).*X.Xpp[2] # only binary materials
    ΣsXmm0 = (s0[1] + s_2[1]).*X.Xmm[1] .+ (s0[2] + s_2[2]).*X.Xmm[2] # only binary materials

    # ΣsXpp_2 = s_2[1] .* X.Xpp[1] .+ s_2[2] .* X.Xpp[2]
    # ΣsXmm_2 = s_2[1] .* X.Xmm[1] .+ s_2[2] .* X.Xmm[2]
    
    ΣτXpp_2 = τ_2[1] .* X.Xpp[1] .+ τ_2[2] .* X.Xpp[2]
    ΣτXmm_2 = τ_2[1] .* X.Xmm[1] .+ τ_2[2] .* X.Xmm[2]
    
    ΣσXpp_2 = σ_2[1] .* X.Xpp[1] .+ σ_2[2] .* X.Xpp[2]
    ΣσXmm_2 = σ_2[1] .* X.Xmm[1] .+ σ_2[2] .* X.Xmm[2]

    AA_pp = (X.∂Xpp ⊗ₓ Ω.∂App .+ ΣτXpp_2 ⊗ Ω.AIpp .- ΣσXpp_2 ⊗Ω.Kpp)
    AA_mm = (ΣτXmm_2 ⊗Ω.AImm .- ΣσXmm_2 ⊗Ω.Kmm)

    AA_mp = X.dXmp ⊗ₓ Ω.dAmp
    AA_pm = X.dXpm ⊗ₓ Ω.dApm

    A_b = [ΣsXpp0 ⊗ Ω.AIpp .- (Δϵ / 2) .* AA_pp     (-(Δϵ / 2)) .* AA_mp
        (Δϵ / 2) .* AA_pm             ΣsXmm0 ⊗Ω.AImm .- (Δϵ / 2) .* AA_mm]

    b = A_b*ψ0 - (Δϵ)*μh((ϵ0 + ϵ) / 2.0)

    A = [ΣsXpp ⊗ Ω.AIpp .+ (Δϵ / 2) .* AA_pp    (Δϵ / 2) .* AA_mp
        (-(Δϵ / 2)) .* AA_pm             ΣsXmm ⊗Ω.AImm .+ (Δϵ / 2) .* AA_mm]
   
    return A, b
end

function solve_forward(solver, X, Ω, physics, (ϵ0, ϵ1), N, gh)
    n_basis = number_of_basis_functions(solver)
    ϵs = range(ϵ1, ϵ0, length=N)
    ψs = [zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)]
    for k in 2:length(ϵs)
        @show k
        A, b = Ab_midpoint(ϵs[k-1], ϵs[k], ψs[k-1], physics, X, Ω, gh)
        ψk = copy(ψs[k-1])
        ψk, log = IterativeSolvers.bicgstabl!(ψk, A, b, 2, log=true, abstol=1e-3, reltol=1e-3)
        @show log
        push!(ψs, ψk)
    end

    return reverse(ϵs), reverse(ψs)
end

function solve_adjoint(solver, X, Ω, physics, (ϵ0, ϵ1), N, μh)
    n_basis = number_of_basis_functions(solver)
    ψs = [zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)]
    ϵs = range(ϵ0, ϵ1, length=N)

    for k in 2:length(ϵs)
        @show k
        A, b = Abx_midpoint(ϵs[k-1], ϵs[k], ψs[k-1], physics, X, Ω, μh)
        ψk = copy(ψs[k-1])
        ψk, log = IterativeSolvers.bicgstabl!(ψk, A, b, 2, log=true)
        @show log
        push!(ψs, ψk)
    end

    return ϵs, ψs
end

function integrate(ϵs, ψs, bh)
    res = 0.0
    Δϵ = ϵs[2] - ϵs[1]
    # b_b = vcat(
    #     bh.x.p⊗bh.Ω.p, 
    #     bh.x.m⊗bh.Ω.m)
    res += 0.5*Δϵ*dot(bh(ϵs[1]), ψs[1])
    for i in 2:length(ϵs)-1
        res += Δϵ*dot(bh(ϵs[i]), ψs[i])
    end
    res += 0.5*Δϵ*dot(bh(ϵs[end]), ψs[end])
    return res
end

function measure_forward(solver, X, Ω, physics, (ϵ0, ϵ1), N, ghs, μhs)
    measurements = zeros(length(ghs.x), length(ghs.Ω), length(ghs.ϵ), length(μhs.x))
    for (i, ghx) ∈ enumerate(ghs.x)
        for (j, ghΩ) ∈ enumerate(ghs.Ω)
            for (k, ghϵ) ∈ enumerate(ghs.ϵ)
                # g_2 = gh.ϵ((ϵ0 + ϵ) / 2.0)
                # b_b = vcat(
                #     gh.x.p⊗gh.Ω.p, 
                #     gh.x.m⊗gh.Ω.m)
                gh(ϵ) = 2*ghϵ(ϵ)*vcat(ghx.p⊗ghΩ.p, ghx.m⊗ghΩ.m)
                ϵs, ψs = solve_forward(solver, X, Ω, physics, (ϵ0, ϵ1), N, gh)
                for (l, (μhx, μhϵ)) ∈ enumerate(zip(μhs.x, μhs.ϵ))
                    μh(ϵ) = μhϵ(ϵ) * vcat(μhx.p ⊗ μhs.Ω[1].p, μhx.m ⊗ μhs.Ω[1].m)
                    measurements[i, j, k, l] = integrate(ϵs, ψs, μh)
                end
            end
        end
    end
    return measurements
end

function measure_adjoint(solver, X, Ω, physics, (ϵ0, ϵ1), N, ghs, μhs, retsol=false)
    measurements = zeros(length(ghs.x), length(ghs.Ω), length(ghs.ϵ), length(μhs.x))
    ψxss = []
    for (l, (μhx, μhϵ)) ∈ enumerate(zip(μhs.x, μhs.ϵ))
        μh(ϵ) = μhϵ(ϵ) * vcat(μhx.p⊗μhs.Ω[1].p, μhx.m⊗μhs.Ω[1].m)
        ϵs, ψxs = solve_adjoint(solver, X, Ω, physics, (ϵ0, ϵ1), N, μh)
        if retsol
            push!(ψxss, ψxs)
        end
        for (i, ghx) ∈ enumerate(ghs.x)
            for (j, ghΩ) ∈ enumerate(ghs.Ω)
                for (k, ghϵ) ∈ enumerate(ghs.ϵ)
                    gh(ϵ) = 2*ghϵ(ϵ)*vcat(ghx.p⊗ghΩ.p, ghx.m⊗ghΩ.m)
                    measurements[i, j, k, l] = integrate(ϵs, ψxs, gh)
                end
            end
        end
    end
    if retsol
        return measurements, ψxss
    end
    return measurements
end

## basis evaluations
function eval_space(U_x, points)
    res = [(spzeros(num_free_dofs(U_x[1])), spzeros(num_free_dofs(U_x[2]))) for _ in points]
    
    funcp = FEFunction(U_x[1], spzeros(num_free_dofs(U_x[1])))
    for k = 1:num_free_dofs(U_x[1])
        funcp.free_values[k] = 1.0
        vals = reshape(evaluate(funcp, points[:]), size(points))
        for i in 1:size(points, 1)
            for j in 1:size(points, 2)
                if !isapprox(vals[i, j], 0.0)
                    res[i, j][1][k] = vals[i, j]
                end
            end
        end
        funcp.free_values[k] = 0.0
    end
    funcm = FEFunction(U_x[2], spzeros(num_free_dofs(U_x[2])))
    for k = 1:num_free_dofs(U_x[2])
        funcm.free_values[k] = 1.0
        vals = reshape(evaluate(funcm, points[:]), size(points))
        for i in 1:size(points, 1)
            for j in 1:size(points, 2)
                if !isapprox(vals[i, j], 0.0)
                    res[i, j][2][k] = vals[i, j]
                end
            end
        end
        funcm.free_values[k] = 0.0
    end
    return res
end



# function project_material_to_grid!(mass_concentrations::NTuple{N, FEFunction}, parametrization::EPMAParametrization, p::Parameters) where N
#     xyz = 

# end