using Gridap
using IterativeSolvers
using LinearAlgebra
using BenchmarkTools
using Pardiso

function center(R)
    return sum(R)/2.0
end

function ⁺(f, Ω0)
    return x -> 1.0/2.0*(f(x) + f(2.0*Ω0-x))
end

function ⁻(f, Ω0)
    return x -> 1.0/2.0*(f(x) - f(2.0*Ω0-x))
end

function ⁺(u::Gridap.MultiField.MultiFieldCellField)
    return u[1]
end

function ⁻(u::Gridap.MultiField.MultiFieldCellField)
    return u[2]
end

function ∂(u::Gridap.CellField)
    return dot(∇(u), VectorValue(1.0))
end

function ∂(u::Gridap.MultiField.MultiFieldCellField)
    return Gridap.MultiField.MultiFieldCellField([∂(⁻(u)), ∂(⁺(u))])
end

import Base.:*
function *(u::Gridap.MultiField.MultiFieldCellField, v::Gridap.MultiField.MultiFieldCellField)
    return Gridap.MultiField.MultiFieldCellField([⁺(u)* ⁺(v) + ⁻(u)* ⁻(v), ⁺(u)* ⁻(v) + ⁻(u)* ⁺(v)])
end

function *(f::Function, u::Gridap.MultiField.MultiFieldCellField)
    Ω0 = get_triangulation(u).model.grid.node_coords[end][1]
    return Gridap.MultiField.MultiFieldCellField([⁺(f, Ω0) * ⁺(u) + ⁻(f, Ω0) * ⁻(u), ⁺(f, Ω0) * ⁻(u) + ⁻(f, Ω0) * ⁺(u)])
end

function EvenOddModel((x₀, x₁), N)
    return CartesianDiscreteModel((x₀, center((x₀, x₁))), (N))
end

function EvenOddFESpace(model, max_order)
    @assert max_order >= 0
    if max_order == 0
        return MultiFieldFESpace([
            TestFESpace(model, ReferenceFE(lagrangian, Float64, max_order+1), conformity=:H1),
            TestFESpace(model, ReferenceFE(lagrangian, Float64, max_order), conformity=:L2)
        ])
    else
        return MultiFieldFESpace([
            TestFESpace(model, ReferenceFE(lagrangian, Float64, max_order), conformity=:H1),
            TestFESpace(model, ReferenceFE(lagrangian, Float64, max_order), conformity=:H1, dirichlet_tags=[2])
        ])
    end
end

function TrialEvenOddFESpace(V, max_order)
    if max_order == 0
        return MultiFieldFESpace([
            TrialFESpace(V[1]), TrialFESpace(V[2])
        ])
    else
        return MultiFieldFESpace([
            TrialFESpace(V[1]), TrialFESpace(V[2], VectorValue(0.0))
        ])
    end
end

function (u::Gridap.MultiField.MultiFieldFEFunction)(x::VectorValue)
    Ω0 = get_triangulation(u).model.grid.node_coords[end][1]
    u1 = Gridap.interpolate(u[1], u[2].fe_space)
    u2 = u[2]
    if x[1] > Ω0
        return u1(-x) - u2(-x)
    else
        return u1(x) + u2(x)
    end
end

import Gridap:interpolate

function interpolate(f, fs::Gridap.MultiFieldFESpace)
    Ω0 = get_triangulation(fs).model.grid.node_coords[end][1]
    vp = Gridap.interpolate(⁺(f, Ω0), fs[1])
    vm = Gridap.interpolate(⁻(f, Ω0), fs[2])
    return Gridap.MultiField.MultiFieldFEFunction([vp.free_values; vm.free_values], fs, [vp, vm])
end

using Plots
using Gridap

R = (-1, 1)
u0 = 1.0
order = 0
N = 16
models = [EvenOddModel(R, n÷2) for n ∈ 2 .^(2:24)]


Ωs = [Triangulation(model) for model in models]
dxs = [Measure(Ω, max(2*order, 2)) for Ω ∈ Ωs]


Vs = [EvenOddFESpace(model, order) for model in models]
using HCubature, SparseArrays

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

function interpolation_matrix((V1, dx1), (V2, dx2))
    A = spzeros(num_free_dofs(V1), num_free_dofs(V2))
    for i in 1:num_free_dofs(V2)
        e_i = zeros(num_free_dofs(V2))
        e_i[i] = 1.0
        v2 = FEFunction(V2, e_i)
        a(u, v) = ∫(⁺(u*v))*dx1
        b(v) = ∫(⁺(v)*(x -> v2[1](x)) + ⁻(v)*(x -> v2[2](x)))*dx1
        op = AffineFEOperator(a, b, TrialEvenOddFESpace(V1, order), V1)
        A[:, i] = Gridap.solve(op).free_values
    end
    return A
end

function smoothing_matrix((V1, dx1), (V2, dx2))
    #A = spzeros(num_free_dofs(V1), num_free_dofs(V2))
    Is = Float64[]
    Js = Float64[]
    Vs = Float64[]
    # @show num_free_dofs(V2)
    for i in 1:num_free_dofs(V2[1])
        push!(Is, 2*(i-1)+1)
        push!(Js, i)
        push!(Vs, 1.0)
        # A[2*(i-1)+1, i] = 1.0
        if 2*(i-1)+1-1 >= 1
            push!(Is, 2*(i-1)+1-1)
            push!(Js, i)
            push!(Vs, 0.5)
            # A[2*(i-1)+1-1, i] = 0.5
        end
        if 2*(i-1)+1+1 < num_free_dofs(V1[1])
            push!(Is, 2*(i-1)+1+1)
            push!(Js, i)
            push!(Vs, 0.5)
            # A[2*(i-1)+1+1, i] = 0.5
        end
    end
    for i in num_free_dofs(V2[1])+1:num_free_dofs(V2)
        push!(Is, 2*(i-1)+1)
        push!(Js, i)
        push!(Vs, 1.0)

        # A[2*(i-1)+1, i] = 1.0
        push!(Is, 2*(i-1)+1-1)
        push!(Js, i)
        push!(Vs, 1.0)
        
        # A[2*(i-1)+1-1, i] = 1.0
    end
    return sparse(Is, Js, Vs)
end

function smoothing_matrix2((V1, dx1), (V2, dx2))
    A = spzeros(num_free_dofs(V1), num_free_dofs(V2))
    a(u, v) = ∫(⁺(u*v))*dx1
    mat = assemble_bilinear(a, (), TrialEvenOddFESpace(V1, order), V1)
    
    # AffineFEOperator(a, b, TrialEvenOddFESpace(V1, order), V1)
    ps = MKLPardisoSolver()
    bs = zeros(num_free_dofs(V1), num_free_dofs(V2))
    v2 = FEFunction(V2, zeros(num_free_dofs(V2)))
    b(v) = ∫(⁺(v)*(x -> v2[1](x)) + ⁻(v)*(x -> v2[2](x)))*dx1

    for i in 1:num_free_dofs(V2)
        @show i, num_free_dofs(V2)
        v2.free_values[i] = 1.0
        bs[:, i] = assemble_linear(b, (), TrialEvenOddFESpace(V1, order), V1)
        v2.free_values[i] = 0.0
    end
    A = sparse(round.(Pardiso.solve(ps, mat, bs), digits=8))
    return A
end


#As = [interpolation_matrix((Vs[i], dxs[i]), (Vs[i+1], dxs[i+1])) for i in 1:length(Vs)-1]
Bs = [smoothing_matrix((Vs[i+1], dxs[i+1]), (Vs[i], dxs[i])) for i in 1:length(Vs)-1]

u2 = interpolate(x -> sin(3*x[1]) + 1.0, Vs[end])
u1 = interpolate(x -> sin(3*x[1]) + 1.0, Vs[end-1])
#u1 = FEFunction(Vs[end-1], As[end]*u2.free_values)
u3 = FEFunction(Vs[end], Bs[end]*u1.free_values)

plotly()
plot(range(R..., length=100), x -> u2(Point(x)))
plot!(range(R..., length=100), x -> u1(Point(x)))
plot!(range(R..., length=100), x -> u3(Point(x)))
plot!(range(R..., length=100), x -> sin(3*x) + 1)

Γ = BoundaryTriangulation(models[end])
dΓ = Measure(Γ, 2*order)

U = TrialEvenOddFESpace(Vs[end], order)
δ⁻(x) = isapprox(x[1], R[1])

# a(u, v) = ∫(∂(⁺(u)) * ⁻(v) - ⁻(u)*∂(⁺(v)) + ⁺(u*v))dxs[end] + ∫(⁺(u)* ⁺(v)*δ⁻)dΓ
a((up, um), (vp, vm)) = ∫(-∂(up) * vm - um*∂(vp) + up*vp - um*vm)dxs[end] + ∫(up * vp*δ⁻)dΓ
b(v) = ∫(u0* ⁺(v) *δ⁻)dΓ

op = AffineFEOperator(a, b, U, Vs[end])

M = Gridap.get_matrix(op)
b_ = Gridap.get_vector(op)

Ms = [M]
bs = [b_]
for i in reverse(1:length(Vs)-1)
    push!(Ms, Bs[i]'*Ms[end]*Bs[i])
    push!(bs, Bs[i]'*bs[end])
end
Ms = reverse(Ms)
bs = reverse(bs)



function multigrid()
    us = [zeros(size(Ms[1], 1))]
    for i in 1:length(Vs)
        # if i != 1
        #     p = plot(range(R..., length=100), x -> FEFunction(Vs[i-1], us[i-1])(Point(x)), label="previous")
        # else
        #     p = plot()
        # end
        # p = plot!(range(R..., length=100), x -> FEFunction(Vs[i], us[i])(Point(x)), label="projection")
        # usi = copy(us[i])
        # usi2 = copy(us[i])
        _, log = minres!(us[i], Ms[i], bs[i], log=true, reltol=1e-3)
        # _, log2 = minres!(zeros(size(Ms[i], 2)), Ms[i], bs[i], log=true, reltol=1e-3)
        # p = plot!(range(R..., length=100), x -> FEFunction(Vs[i], us[i])(Point(x)), label="corrected")
        # p = plot!(range(R..., length=100), x -> FEFunction(Vs[i], Ms[i] \ bs[i])(Point(x)), label="direct")
        # p = plot!(range(R..., length=100), x -> exp(-x-1), label="true")
        # p1 = plot(range(R..., length=100), x -> exp(-x-1) - FEFunction(Vs[i], Ms[i] \ bs[i])(Point(x)), label="direct")
        # p1 = plot!(range(R..., length=100), x -> exp(-x-1) - FEFunction(Vs[i], us[i])(Point(x)), label="corrected")
        # display(plot(p, p1))

        @show log
        # @show log2
        # @show sum((us[i] .- Ms[i] \ bs[i]).^2)
        if i != length(Vs)
            push!(us, Bs[i]*us[i])
        end
    end
    return us[end]
end

@benchmark uh_multigrid = multigrid()

uh_minres = minres!(zeros(size(Ms[end], 2)), Ms[end], bs[end], log=true, reltol=1e-3)

using Pardiso
ps = MKLPardisoSolver()

@benchmark uh_pardiso = Pardiso.solve(ps, Ms[end], bs[end])

plot(range(R..., length=100), x -> FEFunction(Vs[end], uh_pardiso)(Point(x)))
plot!(range(R..., length=100), x -> FEFunction(Vs[end], uh_multigrid)(Point(x)))
plot!(range(R..., length=100), x -> exp(-x-1))



Ms[end]*us[end] .- bs[end]

cond(Matrix(inv(Diagonal(diag(Ms[end])))*Ms[end]))
cond(Matrix(Ms[end]))




plot()
for i in 1:length(Vs)
    uh = FEFunction(Vs[i], us[i])
    plot!(range(R..., length=100), x -> uh(Point(x)))
end
plot!()

u = solve(op)
u_lower = FEFunction(Vs[end-2], u_lower)

plot(range(R..., length=100), x -> u_fe(Point(x)))
plot!(range(R..., length=100), x -> u_lower(Point(x)))
plot!(range(R..., length=100), x -> exp(-x-1))

xs = (range(R..., length=N+1) .+ (R[2] - R[1])/(2*N))[1:end-1]
scatter!(xs, x -> u(Point(x)))

xs |> collect