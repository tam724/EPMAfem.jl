
function closest_point(x::VectorValue{D, T}, start, direction) where {D, T}
    return start + max(dot(x - start, direction), zero(T))*direction
end

function distance_to_line_segment(x::VectorValue, start::VectorValue, direction::VectorValue)
    return norm(x - closest_point(x, start, direction))
end

function line_dirac_approximation(::Union{Dimensions._2D, Dimensions._3D}, x, σ, start, direction)
    distance = distance_to_line_segment(x, start, direction)
    if distance > 6*σ
        return 0.0
    else
        return pdf(Normal(0.0, σ), distance)
    end
end

activation(x, σ) = tanh(x / σ)*0.5 + 0.5
function line_dirac_approximation(::Dimensions._1D, x, σ, start, direction)
    return activation(dot(x - start, direction), σ)
end

function estimate_basis_mean_and_radius(::Dimensions._1D, V, dx)
    # estimate the radii of the basis functions V
    rs = assemble_vector(v -> ∫(v)*dx, V) ./ 2.0

    # compute the means of the basis functions V
    gramian_matrix = assemble_matrix((u, v) -> ∫(u*v)*dx, V, TrialFESpace(V))
    zs = gramian_matrix\assemble_vector(v -> ∫(v * (x -> x[1]))*dx, V)
    
    return [(rs[i], VectorValue(zs[i])) for i in 1:num_free_dofs(V)]
end

function estimate_basis_mean_and_radius(::Dimensions._2D, V, dx)
    # estimate the radii of the basis functions V
    vol = assemble_vector(v -> ∫(v)*dx, V)
    rs = 2.0.*sqrt.(vol./π)

    # compute the means of the basis functions V
    gramian_matrix = assemble_matrix((u, v) -> ∫(u*v)*dx, V, TrialFESpace(V))
    zs = gramian_matrix\assemble_vector(v -> ∫(v * (x -> x[1]))*dx, V)
    xs = gramian_matrix\assemble_vector(v -> ∫(v * (x -> x[2]))*dx, V)
    
    return [(rs[i], VectorValue(zs[i], xs[i])) for i in 1:num_free_dofs(V)]
end

function estimate_basis_mean_and_radius(::Dimensions._3D, V, dx)
    @error "not implemented yet"
    # estimate the radii of the basis functions V
    rs = 2.0.*sqrt.(assemble_vector(v -> ∫(v)*dx, V)./π)

    # compute the means of the basis functions V
    gramian_matrix = assemble_matrix((u, v) -> ∫(u*v)*dx, V, TrialFESpace(V))
    zs = gramian_matrix\assemble_vector(v -> ∫(v * (x -> x[1]))*dx, V)
    xs = gramian_matrix\assemble_vector(v -> ∫(v * (x -> x[2]))*dx, V)
    xs = gramian_matrix\assemble_vector(v -> ∫(v * (x -> x[3]))*dx, V)

    return [(rs[i], VectorValue(zs[i], xs[i], ys[i])) for i in 1:num_free_dofs(V)]
end

function line_integral_contribs(dim, V, dx, σ, start, direction)
    g(x) = line_dirac_approximation(dim, x, σ, start, direction)
    v = assemble_vector(v -> ∫(g*v)dx, V)
    return v
end

function compute_line_integral_contribs(dim::Dimensions.SpaceDimensionality, model, Vabs, Vmat, takeoff_direction)
    dx = Measure(Triangulation(model), 4)

    ps = estimate_basis_mean_and_radius(dim, Vabs, dx)

    g(x, (σ, start)) = line_dirac_approximation(dim, x, σ, start, takeoff_direction)

    # @time test_contribs = SpaceModels.assemble_vectors(g, ps, model, Vmat, dx)

    contribs = zeros(num_free_dofs(Vabs), num_free_dofs(Vmat))
    for i in 1:num_free_dofs(Vabs)
        rs, xs = ps[i]
        contribs[i, :] .= line_integral_contribs(dim, Vmat, dx, rs/2.0, xs, takeoff_direction)
    end

    # @show abs.(contribs .- test_contribs') |> maximum
    # @assert contribs ≈ test_contribs'

    # sparsify the contribs array
    max_contrib = maximum(contribs)
    for i in eachindex(contribs)
        if (contribs[i] / max_contrib) < 1e-10
            contribs[i] = 0.0
        end
    end
    # contribs[contribs ./ max_contrib .< 1e-10] .= 0
    contribs = SparseArrays.dropzeros!(sparse(contribs))
    return contribs
end

function compute_line_integral_contribs(space_model::SpaceModels.GridapSpaceModel, takeoff_direction)
    # we approximate the absorbtion constant cellwise (same as the material)
    Vmat = SpaceModels.material(space_model)
    Vabs = Vmat

    model = space_model.discrete_model
    return compute_line_integral_contribs(SpaceModels.dimensionality(space_model), model, Vabs, Vmat, takeoff_direction)
end

@concrete struct PNNoAbsorption
    model
    arch

    ρ_proj
    element_index
end

function update_bxp!(bxp, updater::PNNoAbsorption, ρs)
    bxp .= updater.ρ_proj*@view(ρs[updater.element_index, :]) |> updater.arch
end

function update_bxp_adjoint!(ρ_adjoint, updater::PNNoAbsorption, bxp_adjoint, ρs)
    @show bxp_adjoint |> size, updater.ρ_proj |> size, ρ_adjoint |> size

    @show transpose(updater.ρ_proj) * (bxp_adjoint |> collect) |> size
    @show @view(ρ_adjoint[updater.element_index, :]) |> size 
    mul!(@view(ρ_adjoint[updater.element_index, :]), transpose(updater.ρ_proj), bxp_adjoint |> collect, true, true)
end

@concrete struct PNAbsorption
    model
    arch

    ρ_proj
    line_contribs
    MAC
    element_index
end

function update_bxp!(bxp, updater::PNAbsorption, ρs)
    bxp .= updater.ρ_proj*(@view(ρs[updater.element_index, :]) .* exp.(.-updater.line_contribs*transpose(ρs)*updater.MAC)) |> updater.arch
end

function update_bxp_adjoint!(ρs_adjoint, updater::PNAbsorption, bxp_adjoint, ρs)
    # too lazy to hand code this ..
    _, back = Zygote.pullback(ρs_ -> updater.ρ_proj*(@view(ρs_[updater.element_index, :]) .* exp.(.-updater.line_contribs*transpose(ρs_)*updater.MAC)), ρs)
    ρs_adjoint .+= back(bxp_adjoint |> collect)[1]
end

function absorption_approximation(updater::PNAbsorption, ρs)
    V = SpaceModels.material(space_model(updater.model))
    return FEFunction(V, exp.(.-updater.line_contribs*transpose(ρs)*updater.MAC))
end
