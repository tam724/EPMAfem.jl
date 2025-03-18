@concrete struct GridapSpaceModel{ND}
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

# dirac basis evaluation
function eval_basis(model, x::Point{D}) where D
    @assert length(dimensions(dimensionality(model))) == D
    δ = DiracDelta(model.discrete_model, x)
    bp = assemble_linear((v, args...) -> δ(v), model, even(model))
    bm = assemble_linear((v, args...) -> δ(v), model, odd(model))
    return (p=bp, m=bm)
end

# dirac basis evaluation
function eval_basis(model, μ::Function)
    bp = assemble_linear(∫R_μv(μ), model, even(model))
    bm = assemble_linear(∫R_μv(μ), model, odd(model))
    return (p=bp, m=bm)
end

# boundary basis evaluation
function eval_basis_boundary(model, μ::Function, dim)
    bp = assemble_linear(∫∂R_ngv{dim}(μ), model, even(model))
    bm = zeros(num_free_dofs(odd(model)))
    return (p=bp, m=bm)
end

function interpolable(vec, model)
    interp = uncached_interpolable(vec, model)
    rand_point = cartesian_unit_vector(Z(), dimensionality(model))
    cache = Gridap.Arrays.return_cache(interp, rand_point)
    return x -> Gridap.Arrays.evaluate!(cache, interp, x)
end

function uncached_interpolable(vec, model)
    if hasproperty(vec, :p) && hasproperty(vec, :m)
        p_func = FEFunction(even(model), Float64.(vec.p))
        m_func = FEFunction(odd(model), Float64.(vec.m))
        return p_func + m_func
    elseif hasproperty(vec, :p)
        p_func = FEFunction(even(model), Float64.(vec.p))
        return p_func
    else
        p_func = FEFunction(even(model), Float64.(vec))
        return p_func
    end
end

