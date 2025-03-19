@concrete struct GridapSpaceModel{ND}
    discrete_model
    even_fe_space
    odd_fe_space
end

function GridapSpaceModel(discrete_model::DiscreteModel{ND, ND}; even=(order=1, conformity=:H1), odd=(order=0, conformity=:L2)) where ND
    reffe1 = ReferenceFE(lagrangian, Float64, even.order)
    even_fe_space = TestFESpace(discrete_model, reffe1, conformity=even.conformity)
    reffe0 = ReferenceFE(lagrangian, Float64, odd.order)
    odd_fe_space = TestFESpace(discrete_model, reffe0, conformity=odd.conformity)
    return GridapSpaceModel{ND}(discrete_model, even_fe_space, odd_fe_space)
end

dimensionality(::GridapSpaceModel{ND}) where ND = dimensionality_type(ND)

boundary_tags(::Dimensions._1D) = (1, 2)
boundary_tag(::Dimensions.Z, ::Dimensions.LeftBoundary, ::Dimensions._1D) = 1
boundary_tag(::Dimensions.Z, ::Dimensions.RightBoundary, ::Dimensions._1D) = 2

boundary_tags(::Dimensions._2D) = (5, 6, 7, 8)
boundary_tag(::Dimensions.Z, ::Dimensions.LeftBoundary, ::Dimensions._2D) = 7
boundary_tag(::Dimensions.Z, ::Dimensions.RightBoundary, ::Dimensions._2D) = 8
boundary_tag(::Dimensions.X, ::Dimensions.LeftBoundary, ::Dimensions._2D) = 5
boundary_tag(::Dimensions.X, ::Dimensions.RightBoundary, ::Dimensions._2D) = 6

boundary_tags(::Dimensions._3D) = (21, 22, 23, 24, 25, 26)
boundary_tag(::Dimensions.Z, ::Dimensions.LeftBoundary, ::Dimensions._3D) = 25
boundary_tag(::Dimensions.Z, ::Dimensions.RightBoundary, ::Dimensions._3D) = 26
boundary_tag(::Dimensions.X, ::Dimensions.LeftBoundary, ::Dimensions._3D) = 23
boundary_tag(::Dimensions.X, ::Dimensions.RightBoundary, ::Dimensions._3D) = 24
boundary_tag(::Dimensions.Y, ::Dimensions.LeftBoundary, ::Dimensions._3D) = 21
boundary_tag(::Dimensions.Y, ::Dimensions.RightBoundary, ::Dimensions._3D) = 22

function get_args(model::GridapSpaceModel)
    dims = dimensionality(model)
    R = Triangulation(model.discrete_model)
    Γ = BoundaryTriangulation(model.discrete_model)
    dx = Measure(R, 6)
    dΓ = Measure(Γ, 6)
    dΓi = Dict((tag => Measure(BoundaryTriangulation(model.discrete_model; tags=tag), 6)) for tag in boundary_tags(dimensionality(model)))
    n = get_normal_vector(Γ)
    return (dims, dx, dΓ, dΓi, n)
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

function projection_fefunc(f, model, space)
    (dims, dx, dΓ, dΓi, n) = get_args(model)
    V = space
    U = TrialFESpace(V)
    op = AffineFEOperator((u, v) -> ∫(u*v)dx, v -> ∫(v*f)dx, U, V)
    return Gridap.solve(op)
end

function projection(f, model, space)
    return projection_fefunc(f, model, space).free_values
end

function L2_projection(f, model)
    projection(f, model, material(model))
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

