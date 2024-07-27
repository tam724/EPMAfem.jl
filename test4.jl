
using Gridap

function assemble_bilinear(a, args, U, V)
    u = get_trial_fe_basis(U)
    v = get_fe_basis(V)
    matcontribs = a(u, v, args...)
    data = Gridap.FESpaces.collect_cell_matrix(U, V, matcontribs)
    return assemble_matrix(SparseMatrixAssembler(U, V), data)
end

model = CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0), (10, 10))

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

pn_sys = build_solver(model, 11, 2)

function mass_concentration(x)
    return x[1]*x[2]
end

function project_function(U, (model, R, dx, ∂R, dΓ, n), f)
	op = AffineFEOperator((u, v) -> ∫(u*v)dx, v -> ∫(v*f)dx, U, U)
	return Gridap.solve(op)
end

∫ρuv(u, v, ρ, (model, R, dx, ∂R, dΓ, n)) = ∫(ρ*u*v)dx

function build_ρ_to_ρp_projection(pn_sys)
    f = project_function(pn_sys.U[2], pn_sys.model, x -> 0.0)
    I = Int32[]
    J = Int32[]
    V = Float64[]
    # iterate unit vectors for f
    for i in 1:num_cells(f)
        fill!(f.free_values, 0.0)
        f.free_values[i] = 1.0
        # assemble full matrix
        mat = assemble_bilinear(∫ρuv, (f, pn_sys.model), pn_sys.U[1], pn_sys.V[1])
        # extract sparsity in the nzvals of the full matrix, v is a sparse vector
        v = sparse(mat.nzval)
        # add this to the "new matrix"
        for j in 1:length(v.nzval)
            push!(I, v.nzind[j])
            push!(J, i)
            push!(V, v.nzval[j])
        end
    end
    return sparse(I, J, V)
end

function build_ρ_to_ρm_projection(pn_sys)
    f = project_function(pn_sys.U[2], pn_sys.model, x -> 1.0)
    # mat is diagonal anyways..
    mat = assemble_bilinear(∫ρuv, (f, pn_sys.model), pn_sys.U[2], pn_sys.V[2])
    return Diagonal(Vector(diag(mat)))
end

using InteractiveUtils
A = build_ρ_to_ρp_projection(pn_sys)
A = build_ρ_to_ρm_projection(pn_sys)

A.nzval |> unique

f = project_function(pn_sys.U[2], pn_sys.model, x -> 0.0)
f.free_values .= rand(100)

@time mat = assemble_bilinear(∫ρuv, (f, pn_sys.model), U1, V1)

maximum(abs.(mat.nzval .- A*f.free_values))

@time A*f.free_values