@concrete struct EPMADetector
    k_ratio
    takeoff_direction
end

@concrete struct EPMAEquations <: EPMAfem.AbstractPNEquations
    elements
    detectors

    dim_basis
    scattering_approx
    energy_model_dimless
    elastic_scattering_cross_section
    bethe_energy_loss
end

function epma_equations(elements, detectors, energy_model_units, PN_N; elastic_scattering_cross_section=NeXLCore.Liljequist1989, bethe_energy_loss=NeXLCore.JoyLuo, ϵ_rel=1e-6)
    energy_interval = energy_model_units[end] - energy_model_units[1]
    max_spatial_range = uconvert(u"nm", maximum(e -> NeXLCore.range(Kanaya1972, NeXLCore.pure(e), ustrip(uconvert(u"eV", energy_model_units[end])), true), elements)u"cm")
    @show max_spatial_range
    dim_basis = DimBasis(max_spatial_range, energy_interval, minimum(e.density for e in elements))

    energy_model_dimless = dimless(energy_model_units, dim_basis)
    
    ## process scattering cross section
    As = [zeros(length(energy_model_units), PN_N+1) for e in 1:length(elements)]
    for (e, elm) in enumerate(elements)
        for (i_n, n) in enumerate(0:PN_N)
            for (i_ϵ, ϵ) in enumerate(energy_model_units)
                ϵ_eV = ϵ / u"eV" |> upreferred
                As[e][i_ϵ, i_n] = dimless(hquadrature(μ -> NeXLCore.δσδΩ(elastic_scattering_cross_section, acos(μ), elm, Float64(ϵ_eV))*Pl.(μ, n), -1, 1, maxevals=100000)[1]*u"cm"^2 / elm.atomic_mass, dim_basis)
            end
        end
    end
    rank, SVDs = compute_svd_approx(As, ϵ_rel)
    EKs = truncate_svd_and_normalize.(SVDs, rank)
    for (e, (E, K)) in enumerate(EKs)
        @show norm(E*K .- As[e]), ϵ_rel*norm(As[e]), norm(As[e])
        @assert norm(E*K .- As[e]) <= ϵ_rel*norm(As[e])
    end
    scattering_approx = [build_scattering_approximation(E, K, energy_model_dimless, rank) for (E, K) in EKs]

    return EPMAEquations(elements, detectors, dim_basis, scattering_approx, energy_model_dimless, elastic_scattering_cross_section, bethe_energy_loss)
end

function EPMAfem.number_of_elements(eq::EPMAEquations)
    return length(eq.elements)
end

function EPMAfem.number_of_scatterings(eq::EPMAEquations)
    length(eq.scattering_approx[1])
end

function stopping_power_units(eq::EPMAEquations, e, ϵ)
    ϵ_eV = ϵ / u"eV" |> upreferred # convert energy to eV
    s = -NeXLCore.dEds(eq.bethe_energy_loss, ϵ_eV, eq.elements[e], 1.0)*u"eV"*u"cm"^2/u"g"
    return s
end

function EPMAfem.stopping_power(eq::EPMAEquations, e, ϵ)
    s = stopping_power_units(eq, e, dimful(ϵ, u"eV", eq.dim_basis))
    return dimless(s, eq.dim_basis)
end

# function EPMAfem.stopping_power(eq::EPMAEquations, energy_model)
#     n_elem = EPMAfem.number_of_elements(eq)
#     S = zeros(n_elem, length(energy_model))
#     for e in 1:n_elem
#         for (i_ϵ, ϵ) in enumerate(energy_model)
#             # s = stopping_power_units(eq, e, dimful(ϵ, u"eV", eq.dim_basis))
#             S[e, i_ϵ] = EPMAfem.stopping_power(eq, e, ϵ)
#         end
#     end
#     return S
# end

function δσδΩ_recon(eq::EPMAEquations, μ, e, ϵ)
    val = 0.0
    for i_s in 1:EPMAfem.number_of_scatterings(eq)
        val += eq.scattering_approx[e][i_s][1](ϵ)*eq.scattering_approx[e][i_s][2](μ)
    end
    return val
end

function σₜ_recon(eq::EPMAEquations, e, ϵ)
    val = 0.0
    for i_s in 1:EPMAfem.number_of_scatterings(eq)
        val += eq.scattering_approx[e][i_s][1](ϵ)
    end
    return val
end

# function EPMAfem.absorption_coefficient(eq::EPMAEquations, energy_model)
#     n_elem = EPMAfem.number_of_elements(eq)
#     A = zeros(n_elem, length(energy_model))
#     for e in 1:n_elem
#         for (i_ϵ, ϵ) in enumerate(energy_model)
#             for i_s = 1:EPMAfem.number_of_scatterings(eq)
#                 # τ = σ
#                 A[e, i_ϵ] += EPMAfem.scattering_coefficient(eq, e, i_s, ϵ)
#             end
#         end
#     end
#     return A
# end

function EPMAfem.absorption_coefficient(eq::EPMAEquations, e, ϵ)
    return sum(EPMAfem.scattering_coefficient(eq, e, i, ϵ) for i in 1:EPMAfem.number_of_scatterings(eq))
end

function EPMAfem.scattering_coefficient(eq::EPMAEquations, e, i, ϵ)
    return eq.scattering_approx[e][i][1](ϵ)
end

# function EPMAfem.scattering_coefficient(eq::EPMAEquations, energy_model)
#     n_elem = EPMAfem.number_of_elements(eq)
#     n_scat = EPMAfem.number_of_scatterings(eq)
#     S = zeros(n_elem, n_scat, length(energy_model))
#     for e in 1:n_elem
#         for (i_ϵ, ϵ) in enumerate(energy_model)
#             for i_s = 1:n_scat
#                 # τ = σ
#                 S[e, i_s, i_ϵ] = EPMAfem.scattering_coefficient(eq, e, i_s, ϵ)
#             end
#         end
#     end
#     return S
# end

function EPMAfem.mass_concentrations(eq::EPMAEquations, e, x)
    return e == 1 ? dimless(eq.elements[1].density, eq.dim_basis) : 0.0
end

function EPMAfem.scattering_kernel(eq::EPMAEquations, e, i)
    return eq.scattering_approx[e][i][2]
end

## problem discretization uses the main discretization funciton @see ./src/pndiscretization.jl


## Because for EPMA the equations are more or less tied to the extraction and the boundary conditions, we use EPMAEquations for everything:

## Extractions

EPMAfem.number_of_extractions(eq::EPMAEquations) = length(eq.detectors)

function EPMAfem.extraction_energy_distribution(eq::EPMAEquations, i, ϵ)
    ϵ_dim = dimful(ϵ, u"eV", eq.dim_basis)
    return dimless(NeXLCore.ionizationcrosssection(eq.detectors[i].k_ratio |> NeXLCore.inner, ustrip(ϵ_dim))*u"cm^2", eq.dim_basis)
end

function mass_absorption_coefficient(eq::EPMAEquations, k_ratio)
    return [dimless(NeXLCore.mac(elm, k_ratio)u"cm^2/g", eq.dim_basis) for elm in eq.elements]
end

function element_index(eq, k_ratio)
    return only(findall(e -> element(k_ratio) == e, eq.elements))
end

function discretize_detectors(eq::EPMAEquations, mdl::EPMAfem.DiscretePNModel, arch::EPMAfem.PNArchitecture; absorption=true, updatable=true)
    T = EPMAfem.base_type(arch)

    ## instantiate Gridap
    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels

    space_mdl = EPMAfem.space_model(mdl)
    direction_mdl = EPMAfem.direction_model(mdl)

    μϵs = [Vector{T}([EPMAfem.extraction_energy_distribution(eq, i, ϵ) for ϵ ∈ EPMAfem.energy_model(mdl)]) for i in 1:EPMAfem.number_of_extractions(eq)]
    # isotropic in direction
    nb = EPMAfem.n_basis(mdl)
    @assert SphericalHarmonicsModels.eo(direction_mdl) == :OE
    μΩp = zeros(nb.nΩ.p) |> arch
    μΩm = SH.assemble_linear(SH.∫S²_hv(Ω -> 1.0), direction_mdl, SH.minus(direction_mdl)) |> arch

    ρs = EPMAfem.discretize_mass_concentrations(eq, mdl)
    ρ_proj = nothing #SM.assemble_bilinear(SM.∫R_uv, space_mdl, SM.odd(space_mdl), SM.even(space_mdl))
    n_parameters = (EPMAfem.number_of_elements(eq), EPMAfem.n_basis(mdl).nx.m)

    # only compute the line_contribs for unique takeoff directions
    unique_takeoff_directions = unique([det.takeoff_direction for det in eq.detectors])
    @show length(unique_takeoff_directions)
    if absorption
        line_integral_contribs = [EPMAfem.compute_line_integral_contribs(space_mdl, takeoff_direction) for takeoff_direction in unique_takeoff_directions]
        absorptions = [EPMAfem.PNAbsorption(mdl, arch, ρ_proj, line_integral_contribs[findfirst(tak_dir -> tak_dir == eq.detectors[i].takeoff_direction, unique_takeoff_directions)], mass_absorption_coefficient(eq, eq.detectors[i].k_ratio), element_index(eq, eq.detectors[i].k_ratio)) for i in 1:EPMAfem.number_of_extractions(eq)]
    else
        absorptions = [EPMAfem.PNNoAbsorption(mdl, arch, ρ_proj, element_index(eq, eq.detectors[i].k_ratio)) for i in 1:EPMAfem.number_of_extractions(eq)]
    end

    vecs = [EPMAfem.UpdatableRank1DiscretePNVector(EPMAfem.Rank1DiscretePNVector(true, mdl, arch, μϵs[i], (p=EPMAfem.allocate_vec(arch, EPMAfem.n_basis(mdl).nx.p), m=EPMAfem.allocate_vec(arch, EPMAfem.n_basis(mdl).nx.m)), (p=μΩp, m=μΩm)), absorptions[i], n_parameters) for i in 1:EPMAfem.number_of_extractions(eq)]
    for vec in vecs
        EPMAfem.update_vector!(vec, ρs)
    end
    if updatable
        return vecs
    else
        return [vec.vector for vec in vecs]
    end
    # else
    #     μxps = [SM.assemble_linear(SM.∫R_μv(x -> extraction_space_distribution(pn_ex, i, x)), space_mdl, SM.even(space_mdl)) for i in 1:number_of_extractions(pn_ex)] |> arch
    #     return [Rank1DiscretePNVector(true, mdl, arch, μϵs[i], μxps[i], μΩps[i]) for i in 1:number_of_extractions(pn_ex)]
    # end
end

