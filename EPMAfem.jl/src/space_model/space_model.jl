abstract type AbstractSpaceModel{ND} end
@concrete struct GridapSpaceModel{ND} <: AbstractSpaceModel{ND}
    discrete_model
end

function GridapSpaceModel(discrete_model::DiscreteModel{ND, ND}) where ND
    return GridapSpaceModel{ND}(discrete_model)
end

dimensionality(model::GridapSpaceModel{ND}) where ND = dimensionality_type(ND)

function get_args(model::GridapSpaceModel)
    dims = dimensionality(model)
    R = Triangulation(model.discrete_model)
    Γ = BoundaryTriangulation(model.discrete_model)
    dx = Measure(R, 6)
    dΓ = Measure(Γ, 6)
    n = get_normal_vector(Γ)
    return (dims, R, dx, Γ, dΓ, n)
end

function even(model::GridapSpaceModel)
    reffe = ReferenceFE(lagrangian, Float64, 1)
    return TestFESpace(model.discrete_model, reffe, conformity=:H1)
end

function odd(model::GridapSpaceModel)
    reffe = ReferenceFE(lagrangian, Float64, 0)
    return TestFESpace(model.discrete_model, reffe, conformity=:L2)
end

function material(model::GridapSpaceModel)
    reffe = ReferenceFE(lagrangian, Float64, 0)
    return TestFESpace(model.discrete_model, reffe, conformity=:L2)
end

function n_basis(model::GridapSpaceModel)
    return (p=num_free_dofs(even(model)), m=num_free_dofs(odd(model)))
end

function L2_projection(f, model)
    (dims, R, dx, Γ, dΓ, n) = get_args(model)
    V = material(model)
    U = TrialFESpace(V)
    op = AffineFEOperator((u, v) -> ∫(u*v)dx, v -> ∫(v*f)dx, U, V)
    return Gridap.solve(op).free_values
end