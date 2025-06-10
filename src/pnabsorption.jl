
function closest_point(x, start, direction)
    t = dot(x - start, direction)
    if t < 0
        return start
    else
        return start + t * direction
    end
end

function distance_to_line_segment(x::VectorValue, start::VectorValue, direction::VectorValue)
    return norm(x - closest_point(x, start, direction))
end

function line_dirac_approximation(x, σ, start, direction)
    distance = distance_to_line_segment(x, start, direction)
    if distance > 6*σ
        return 0.0
    else
        return pdf(Normal(0.0, σ), distance)
    end
end

function line_integral_contribs(V, dx, σ, start, direction)
    g(x) = line_dirac_approximation(x, σ, start, direction)
    v = assemble_vector(v -> ∫(g*v)dx, V)
    return v
end

function estimate_basis_mean_and_radius(V, dx)
    # estimate the radii of the basis functions V
    rs = 2.0.*sqrt.(assemble_vector(v -> ∫(v)*dx, V)./π)

    # compute the means of the basis functions V
    gramian_matrix = assemble_matrix((u, v) -> ∫(u*v)*dx, V, TrialFESpace(V))
    xs = gramian_matrix\assemble_vector(v -> ∫(v * (x -> x[1]))*dx, V)
    ys = gramian_matrix\assemble_vector(v -> ∫(v * (x -> x[2]))*dx, V)
    return rs, xs, ys
end

function compute_line_integral_contribs(model, Vabs, Vmat, takeoff_direction)
    dx = Measure(Triangulation(model), 10)

    rs, xs, ys = estimate_basis_mean_and_radius(Vabs, dx)

    contribs = zeros(num_free_dofs(Vabs), num_free_dofs(Vmat))
    for i in 1:num_free_dofs(Vabs)
        contribs[i, :] .= line_integral_contribs(Vmat, dx, rs[i]/2, VectorValue(xs[i], ys[i]), takeoff_direction)
    end

    # sparsify the contribs array
    max_contrib = maximum(contribs)
    contribs[contribs ./ max_contrib .< 1e-10] .= 0
    contribs = SparseArrays.dropzeros!(sparse(contribs))
    return contribs
end

function compute_line_integral_contribs(space_model::SpaceModels.GridapSpaceModel, takeoff_direction)
    # we approximate the absorbtion constant cellwise (same as the material)
    Vmat = SpaceModels.material(space_model)
    Vabs = Vmat

    model = space_model.discrete_model
    return compute_line_integral_contribs(model, Vabs, Vmat, takeoff_direction)
end

@concrete struct PNNoAbsorbtion
    model
    arch

    ρ_proj
    element_index
end

function update_bxp!(bxp, updater::PNNoAbsorbtion, ρs)
    bxp .= updater.ρ_proj*@view(ρs[updater.element_index, :]) |> updater.arch
end

function update_bxp_adjoint!(ρ_adjoint, updater::PNNoAbsorbtion, bxp_adjoint, ρs)
    @show bxp_adjoint |> size, updater.ρ_proj |> size, ρ_adjoint |> size

    @show transpose(updater.ρ_proj) * (bxp_adjoint |> collect) |> size
    @show @view(ρ_adjoint[updater.element_index, :]) |> size 
    mul!(@view(ρ_adjoint[updater.element_index, :]), transpose(updater.ρ_proj), bxp_adjoint |> collect, true, true)
end

@concrete struct PNAbsorbtion
    model
    arch

    ρ_proj
    line_contribs
    MAC
    element_index
end

function update_bxp!(bxp, updater::PNAbsorbtion, ρs)
    bxp .= updater.ρ_proj*(@view(ρs[updater.element_index, :]) .* exp.(.-updater.line_contribs*transpose(ρs)*updater.MAC)) |> updater.arch
end

function update_bxp_adjoint!(ρs_adjoint, updater::PNAbsorbtion, bxp_adjoint, ρs)
    # too lazy to hand code this ..
    _, back = Zygote.pullback(ρs_ -> updater.ρ_proj*(@view(ρs_[updater.element_index, :]) .* exp.(.-updater.line_contribs*transpose(ρs_)*updater.MAC)), ρs)
    ρs_adjoint .+= back(bxp_adjoint |> collect)[1]
end
