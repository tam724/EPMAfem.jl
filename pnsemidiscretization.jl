
using StaticArrays

"""
    DiscretePNProblem

    discrete version of [`PNEquations`](@ref)
"""
struct DiscretePNProblem{T, V<:AbstractVector{T}, M<:AbstractMatrix{T}, SM<:AbstractSparseMatrix{T}, PNM<:PNGridapModel}
    model::PNM

    # energy (these will always live on the cpu)
    s::Matrix{T}
    τ::Matrix{T}
    σ::Array{T, 3}

    # space (might be moved to gpu)
    ρp::Vector{SM}
    ρm::Vector{Diagonal{T, V}}
    ∂p::Vector{SM}
    ∇pm::Vector{SM}

    # direction (might be moved to gpu)
    Ip::Diagonal{T, V}
    Im::Diagonal{T, V}
    kp::Vector{Vector{Diagonal{T, V}}}
    km::Vector{Vector{Diagonal{T, V}}}
    absΩp::Vector{M}
    Ωpm::Vector{M}
end

struct DiscretePNRHS{T, V<:AbstractVector{T}, PNM<:PNGridapModel}
    model::PNM

    weights::Array{T, 3}
    bϵ::Vector{Vector{T}}

    #(might be moved to gpu)
    bxp::Vector{V}
    bΩp::Vector{V}
end

mat_type(pnprob::DiscretePNProblem) = mat_type(pnprob.model)
vec_type(pnprob::DiscretePNProblem) = vec_type(pnprob.model)
base_type(pnprob::DiscretePNProblem) = base_type(pnprob.model)

function discretize_problem(pn_eq::PNEquations, discrete_model::PNGridapModel)
    MT = mat_type(discrete_model)
    VT = vec_type(discrete_model)
    SMT = smat_type(discrete_model)
    T = base_type(discrete_model)

    ϵs = energy(discrete_model)

    n_elem = number_of_elements(pn_eq)
    n_scat = number_of_scatterings(pn_eq)

    ## assemble (compute) all the energy matrices
    s = Matrix{T}([_s(pn_eq, e)(ϵ) for e in 1:n_elem, ϵ ∈ ϵs])
    τ = Matrix{T}([_τ(pn_eq, e)(ϵ) for e in 1:n_elem, ϵ ∈ ϵs])
    σ = Array{T}([_σ(pn_eq, e, i)(ϵ) for e in 1:n_elem, i in 1:n_scat, ϵ ∈ ϵs])

    ## instantiate Gridap stuff
    U, V, gap_model = gridap_model(space(discrete_model))
    
    n_basis = number_of_basis_functions(discrete_model)

    ## assemble all the space matrices
    ρp = [SMT((assemble_bilinear(∫ρuv, (_mass_concentrations(pn_eq, e), gap_model), U[1], V[1]))) for e in 1:number_of_elements(pn_eq)] 
    ρm = [Diagonal(VT(diag(assemble_bilinear(∫ρuv, (_mass_concentrations(pn_eq, e), gap_model), U[2], V[2])))) for e in 1:number_of_elements(pn_eq)] 

    ∂p = [SMT(dropzeros(assemble_bilinear(a, (gap_model, ), U[1], V[1]))) for a ∈ ∫absn_uv(space(discrete_model))]
    ∇pm = [SMT(assemble_bilinear(a, (gap_model, ), U[1], V[2])) for a ∈ ∫∂u_v(space(discrete_model))]

    ## assemble all the direction matrices
    Kpp, Kmm = assemble_scattering_matrices(max_degree(discrete_model), _electron_scattering_kernel(pn_eq, 1, 1), nd(discrete_model))
    Kpp = Diagonal(VT(Kpp.diag))
    Kmm = Diagonal(VT(-Kmm.diag))
    kp = [[Kpp], [Kpp]]
    # we use negative odd basis function for direction (resulting matrix is symmetric)
    km = [[Kmm], [Kmm]]

    Ip = Diagonal(VT(ones(n_basis.Ω.p)))
    # we use negative odd basis function for direction (resulting matrix is symmetric)
    Im = Diagonal(VT(-ones(n_basis.Ω.m)))

    absΩp = [MT(assemble_boundary_matrix(max_degree(discrete_model), dir, :pp, nd(discrete_model), 0.0)) for dir ∈ space_directions(discrete_model)]
    Ωpm = [MT(assemble_transport_matrix(max_degree(discrete_model), dir, :pm, nd(discrete_model))) for dir ∈ space_directions(discrete_model)]

    DiscretePNProblem(discrete_model, s, τ, σ, ρp, ρm, ∂p, ∇pm, Ip, Im, kp, km, absΩp, Ωpm)
end

function discretize_rhs(pn_eq::PNEquations, discrete_model::PNGridapModel)
    VT = vec_type(discrete_model)
    T = base_type(discrete_model)

    ## instantiate Gridap stuff
    U, V, gap_model = gridap_model(space(discrete_model))

    n_basis = number_of_basis_functions(discrete_model)

    ## assemble excitation 
    gϵ = Vector{T}([_excitation_energy_distribution(pn_eq)(ϵ) for ϵ ∈ energy(discrete_model)])
    gxp = VT(assemble_linear(∫nzgv, (_excitation_spatial_distribution(pn_eq), gap_model), U[1], V[1]))
    gΩp = VT(assemble_direction_boundary(max_degree(discrete_model), _excitation_direction_distribution(pn_eq), @SVector[0.0, 0.0, 1.0], nd(discrete_model)).p)

    return DiscretePNRHS(discrete_model, ones(T, 1, 1, 1), [gϵ], [gxp], [gΩp])
end

function discretize_extraction(pn_eq::PNEquations, discrete_model::PNGridapModel)
    VT = vec_type(discrete_model)
    T = base_type(discrete_model)

    ## instantiate Gridap stuff
    U, V, gap_model = gridap_model(space(discrete_model))

    n_basis = number_of_basis_functions(discrete_model)

    ## ... and extraction
    μϵ = Vector{T}([_extraction_energy_distribution(pn_eq)(ϵ) for ϵ ∈ energy(discrete_model)])
    μxp = VT(assemble_linear(∫μv, (_extraction_spatial_distribution(pn_eq), gap_model), U[1], V[1]))
    μΩp = VT(assemble_direction_source(max_degree(discrete_model), _extraction_direction_distribution(pn_eq), nd(discrete_model)).p)

    return DiscretePNRHS(discrete_model, ones(T, 1, 1, 1), [μϵ], [μxp], [μΩp])
end

function discretize(pn_eq::PNEquations, discrete_model::PNGridapModel)
    return discretize_problem(pn_eq, discrete_model), discretize_rhs(pn_eq, discrete_model)
end

function assemble_rhs_p!(b, rhs::DiscretePNRHS, i, Δ; bxp=rhs.bxp, bΩp=rhs.bΩp)
    fill!(b, zero(eltype(b)))

    nLp = length(first(bxp))
    nRp = length(first(bΩp))

    bp = reshape(@view(b[1:nLp*nRp]), (nLp, nRp))
    # bp = pview(b, rhs.model)
    for (ϵi, bϵi) in zip(1:size(rhs.weights, 1), rhs.bϵ)
        for (xi, bxpi) in zip(1:size(rhs.weights, 2), bxp)
            for (Ωi, bΩpi) in zip(1:size(rhs.weights, 3), bΩp)
                bϵ2 = 0.5*(bϵi[i] + bϵi[i+1])
                bxpi_mat = reshape(@view(bxpi[:]), (length(bxpi), 1))
                bΩpi_mat = reshape(@view(bΩpi[:]), (1, length(bΩpi)))
                mul!(bp, bxpi_mat, bΩpi_mat, rhs.weights[ϵi, xi, Ωi]*bϵ2*Δ, 1.0)
            end
        end
    end
end



## keep this for the EPMADiscretization
# function pn_semidiscretization(pn_sys, pn_equ, skip_projections=false)
#     U = pn_sys.U    
#     V = pn_sys.V
#     model = pn_sys.model

#     n_basis = number_of_basis_functions(pn_sys)
#     # TODO!!! (support multiple scattering kernels per element)
#     Kpp, Kmm = assemble_scattering_matrices(pn_sys.PN, _electron_scattering_kernel(pn_equ, 1, 1), nd(pn_sys))

#     ρ_to_ρp = sparse(zeros(1, 1))
#     ρ_to_ρm = Diagonal(zeros(1))
#     ρp = nothing
#     ρm = nothing
#     if !skip_projections
#         ρ_to_ρp = build_ρ_to_ρp_projection(pn_sys)
#         ρ_to_ρm = build_ρ_to_ρm_projection(pn_sys)

#         ρs = [project_function(U[2], model, _mass_concentrations(pn_equ, e)).free_values for e in 1:number_of_elements(pn_equ)]

#         temp = assemble_bilinear(∫ρuv, (x -> 1.0, model), U[1], V[1]) # only get the sparsity pattern of ρp and reuse the colptr and rowval vectors
#         ρp_colptr = Vector(temp.colptr)
#         ρp_rowval = Vector(temp.rowval)
#         ρp = [SparseMatrixCSC{Float64, Int64}(n_basis.x.p, n_basis.x.p, ρp_colptr, ρp_rowval, zeros(Float64, length(temp.nzval))) for _ in 1:number_of_elements(pn_equ)]
#         ρm = [Diagonal(zeros(Float64, n_basis.x.m)) for _ in 1:number_of_elements(pn_equ)] 
#         for e in 1:number_of_elements(pn_equ)
#             mul!(ρp[e].nzval, ρ_to_ρp, ρs[e])
#             mul!(ρm[e].diag, ρ_to_ρm, ρs[e])
#         end
#     else
#         ρp = [assemble_bilinear(∫ρuv, (_mass_concentrations(pn_equ, e), model), U[1], V[1]) for e in 1:number_of_elements(pn_equ)]
#         ρm = [Diagonal(Vector(diag(assemble_bilinear(∫ρuv, (_mass_concentrations(pn_equ, e), model), U[2], V[2])))) for e in 1:number_of_elements(pn_equ)] 
#     end
#     return PNSemidiscretization(
#         ((n_basis.x.p, n_basis.x.m), (n_basis.Ω.p, n_basis.Ω.m)),
#         pn_equ,

#         ρp,
#         ρm,

#         [assemble_bilinear(a, (model, ), U[1], V[1]) for a ∈ ∫absn_uv(model.model)],
#         [assemble_bilinear(a, (model, ), U[1], V[2]) for a ∈ ∫∂u_v(model.model)],

#         Diagonal(ones(n_basis.Ω.p)),
#         # we use negative odd basis function for direction (resulting matrix is symmetric)
#         Diagonal(-ones(n_basis.Ω.m)),

#         [[Kpp], [Kpp]],
#         # we use negative odd basis function for direction (so the resulting matrix is symmetric)
#         [[-Kmm], [-Kmm]],

#         [assemble_boundary_matrix(pn_sys.PN, dir, :pp, nd(model.model)) for dir ∈ space_directions(model.model)],
#         [assemble_transport_matrix(pn_sys.PN, dir, :pm, nd(model.model)) for dir ∈ space_directions(model.model)],

#         # only discretize the even parts (and store as matrices (with 1x or x1 dimension) then CUDA works):
#         [sparse(reshape(assemble_linear(∫nzgv, (_excitation_spatial_distribution(pn_equ, j), model), U[1], V[1]), (n_basis.x.p, 1),)) for j in 1:number_of_beam_positions(pn_equ)],
#         [Matrix(reshape(assemble_direction_boundary(pn_sys.PN, _excitation_direction_distribution(pn_equ, k), @SVector[0.0, 0.0, 1.0], nd(model.model)).p, (1, n_basis.Ω.p))) for k in 1:number_of_beam_directions(pn_equ)],

#         [Matrix(reshape(assemble_linear(∫μv, (_extraction_spatial_distribution(pn_equ, e), model), U[1], V[1]), (n_basis.x.p, 1))) for e in 1:number_of_extraction_positions(pn_equ)],
#         [Matrix(reshape(assemble_direction_source(pn_sys.PN, _extraction_direction_distribution(pn_equ, k), nd(model.model)).p, (1, n_basis.Ω.p))) for k in 1:number_of_extraction_directions(pn_equ)],

#         ρ_to_ρp,
#         ρ_to_ρm
#     )
# end



# function update_mass_concentrations!(pn_semi, ρs)
#     for e in 1:number_of_elements(equations(pn_semi))
#         mul!(pn_semi.ρp[e].nzval, pn_semi.ρ_to_ρp, ρs[e])
#         mul!(pn_semi.ρm[e].diag, pn_semi.ρ_to_ρm, ρs[e])

#         # also update the μs
#     end
# end

function assemble_rhs_hightolow!(b, problem::DiscretePNProblem, gϵ, α)
    bp = pview(b, problem.model)
    bm = mview(b, problem.model)

    mul!(bp, problem.gx, problem.gΩ, α*gϵ, false)
    fill!(bm, zero(eltype(b)))
end

# function assemble_extraction_rhs!(b, pn_semi::PNSemidiscretization, ϵ, (i, j, k), α)
#     ((nLp, nLm), (nRp, nRm)) = pn_semi.size
#     np = nLp*nRp
#     nm = nLm*nRm

#     bp = reshape(@view(b[1:np]), (nLp, nRp))
#     bm = reshape(@view(b[np+1:np+nm]), (nLm, nRm))

#     mul!(bp, pn_semi.μx[j], pn_semi.μΩ[k], α*_extraction_energy_distribution(pn_semi.pn_equ, i)(ϵ), false)
#     fill!(bm, 0)
# end

function project_function(U, (model, R, dx, ∂R, dΓ, n), f)
    op = AffineFEOperator((u, v) -> ∫(u*v)dx, v -> ∫(v*f)dx, U, U)
    return Gridap.solve(op)
end


function build_ρ_to_ρp_projection(pn_sys)
    f = FEFunction(pn_sys.U[2], zeros(num_free_dofs(pn_sys.U[2])))
    I = Int64[]
    J = Int64[]
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