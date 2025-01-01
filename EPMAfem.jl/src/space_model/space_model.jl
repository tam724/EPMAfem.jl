abstract type AbstractSpaceModel{ND} end
@concrete struct GridapSpaceModel{ND} <: AbstractSpaceModel{ND}
    discrete_model
    even_fe_space
    odd_fe_space
end

function GridapSpaceModel(discrete_model::DiscreteModel{ND, ND}) where ND
    reffe1 = ReferenceFE(lagrangian, Float64, 1)
    even_fe_space = TestFESpace(discrete_model, reffe1, conformity=:H1)
    reffe0 = ReferenceFE(lagrangian, Float64, 0)
    odd_fe_space = TestFESpace(discrete_model, reffe0, conformity=:L2)
    return GridapSpaceModel{ND}(discrete_model, even_fe_space, odd_fe_space)
end

dimensionality(::GridapSpaceModel{ND}) where ND = dimensionality_type(ND)

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
    return model.even_fe_space
end

function odd(model::GridapSpaceModel)
    return model.odd_fe_space
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

function projection(f, model, space)
    (dims, R, dx, Γ, dΓ, n) = get_args(model)
    V = space
    U = TrialFESpace(V)
    op = AffineFEOperator((u, v) -> ∫(u*v)dx, v -> ∫(v*f)dx, U, V)
    return Gridap.solve(op).free_values
end