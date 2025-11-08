abstract type GridapSpaceModel{ND} end
@concrete struct GeneralSpaceModel{ND} <: GridapSpaceModel{ND}
    discrete_model
    plus_fe_space
    minus_fe_space
    args
end
@concrete struct CartesianSpaceModel{ND} <: GridapSpaceModel{ND}
    discrete_model
    plus_fe_space
    minus_fe_space
    args

    discrete_models
    plus_fe_spaces
    minus_fe_spaces
    _args
end

function GeneralSpaceModel(discrete_model::DiscreteModel{ND, ND}; plus=(name=lagrangian, order=1, conformity=:H1), minus=(name=lagrangian, order=0, conformity=:L2)) where ND
    reffe1 = ReferenceFE(plus.name, Float64, plus.order)
    plus_fe_space = TestFESpace(discrete_model, reffe1, conformity=plus.conformity)
    reffe0 = ReferenceFE(minus.name, Float64, minus.order)
    minus_fe_space = TestFESpace(discrete_model, reffe0, conformity=minus.conformity)
    return GeneralSpaceModel{ND}(discrete_model, plus_fe_space, minus_fe_space, get_args_(discrete_model))
end

function GridapSpaceModel(discrete_model::DiscreteModel; plus=(order=1, conformity=:H1), minus=(order=0, conformity=:L2))
   return GeneralSpaceModel(discrete_model, plus=plus, minus=minus)
end

function GridapSpaceModel(discrete_model::CartesianDiscreteModel{ND}; plus=(name=lagrangian, order=1, conformity=:H1), minus=(name=lagrangian, order=0, conformity=:L2)) where ND
    c_descr = Gridap.Geometry.get_cartesian_descriptor(discrete_model)
    c_descrs = Gridap.Geometry.CartesianDescriptor.(VectorValue.(c_descr.origin.data), tuple.(c_descr.sizes), c_descr.partition)
    discrete_models = CartesianDiscreteModel.(c_descrs)

    plus_reffe = ReferenceFE(plus.name, Float64, plus.order)
    plus_fe_space = TestFESpace(discrete_model, plus_reffe, conformity=plus.conformity)
    plus_fe_spaces = TestFESpace.(discrete_models, Ref(plus_reffe), conformity=plus.conformity)
    
    minus_reffe = ReferenceFE(minus.name, Float64, minus.order)
    minus_fe_space = TestFESpace(discrete_model, minus_reffe, conformity=minus.conformity)
    minus_fe_spaces = TestFESpace.(discrete_models, Ref(minus_reffe), conformity=minus.conformity)

    return CartesianSpaceModel{ND}(discrete_model, plus_fe_space, minus_fe_space, get_args_(discrete_model), discrete_models, plus_fe_spaces, minus_fe_spaces, get_args_.(discrete_models))
end

Dimensions.dimensionality(::GridapSpaceModel{ND}) where ND = dimensionality(ND)

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

function get_args_(discrete_model::DiscreteModel{ND, ND}) where ND
    dims = dimensionality(ND)
    R = Triangulation(discrete_model)
    Γ = BoundaryTriangulation(discrete_model)
    dx = Measure(R, 6)
    dΓ = Measure(Γ, 6)
    dΓi = Dict((tag => Measure(BoundaryTriangulation(discrete_model; tags=tag), 6)) for tag in boundary_tags(dims))
    n = get_normal_vector(Γ)
    return (dims, dx, dΓ, dΓi, n)
end

get_args(model::GridapSpaceModel) = model.args

function plus(model::GridapSpaceModel)
    return model.plus_fe_space
end

function minus(model::GridapSpaceModel)
    return model.minus_fe_space
end

function material(model::GridapSpaceModel)
    reffe = ReferenceFE(lagrangian, Float64, 0)
    return TestFESpace(model.discrete_model, reffe, conformity=:L2)
end

function n_basis(model::GridapSpaceModel)
    return (p=num_free_dofs(plus(model)), m=num_free_dofs(minus(model)))
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
    bp = assemble_linear((v, args...) -> δ(v), model, plus(model))
    bm = assemble_linear((v, args...) -> δ(v), model, minus(model))
    return (p=bp, m=bm)
end

# dirac basis evaluation
function eval_basis(model, μ::Function)
    bp = assemble_linear(∫R_μv(μ), model, plus(model))
    bm = assemble_linear(∫R_μv(μ), model, minus(model))
    return (p=bp, m=bm)
end

# boundary basis evaluation
function eval_basis_boundary(model, μ::Function, dim)
    bp = assemble_linear(∫∂R_ngv{dim}(μ), model, plus(model))
    bm = zeros(num_free_dofs(minus(model)))
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
        p_func = FEFunction(plus(model), Float64.(vec.p))
        m_func = FEFunction(minus(model), Float64.(vec.m))
        return p_func + m_func
    elseif hasproperty(vec, :p)
        p_func = FEFunction(plus(model), Float64.(vec.p))
        return p_func
    else
        p_func = FEFunction(plus(model), Float64.(vec))
        return p_func
    end
end

