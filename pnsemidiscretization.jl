
using StaticArrays

"""
PNSemidiscretization.
Basically a matrix storage, that has some logic to assemble the decomposed rhs (g and μ) (for given energy ϵ)
"""
struct PNSemidiscretization{T, V<:AbstractVector{T}, M<:AbstractMatrix{T}, SM<:AbstractSparseMatrix{T}, Teq}
    size::Tuple{Tuple{Int64, Int64}, Tuple{Int64, Int64}}
    pn_equ::Teq

    ρp::Vector{SM}
    ρm::Vector{Diagonal{T, V}}
    ∂p::Vector{SM}
    ∇pm::Vector{SM}

    Ip::Diagonal{T, V}
    Im::Diagonal{T, V}

    kp::Vector{Vector{Diagonal{T, V}}}
    km::Vector{Vector{Diagonal{T, V}}}

    absΩp::Vector{M}
    Ωpm::Vector{M}

    # For source (g) and extraction (μ) we only discretize the even parts (the odd parts should always be zero.)
    gx::Vector{SM}
    gΩ::Vector{M}

    μx::Vector{M}
    μΩ::Vector{M}

    ρ_to_ρp::SM
    ρ_to_ρm::Diagonal{T, V}
end

function cuda(pn_semi::PNSemidiscretization) # TODO: add type argument
    return PNSemidiscretization(
        pn_semi.size,
        pn_semi.pn_equ,

        cu.(pn_semi.ρp),
        cu.(pn_semi.ρm),
        cu.(pn_semi.∂p),
        cu.(pn_semi.∇pm),

        cu(pn_semi.Ip),
        cu(pn_semi.Im),

        [cu.(kpe) for kpe in pn_semi.kp],
        [cu.(kme) for kme in pn_semi.km],

        cu.(pn_semi.absΩp),
        cu.(pn_semi.Ωpm),

        cu.(pn_semi.gx),
        cu.(pn_semi.gΩ),

        cu.(pn_semi.μx),
        cu.(pn_semi.μΩ),

        cu(pn_semi.ρ_to_ρp),
        cu(pn_semi.ρ_to_ρm),
    )
end

function pn_semidiscretization(pn_sys, pn_equ)
    U = pn_sys.U    
    V = pn_sys.V
    model = pn_sys.model

    n_basis = number_of_basis_functions(pn_sys)
    # TODO!!! (support multiple scattering kernels per element)
    Kpp, Kmm = assemble_scattering_matrices(pn_sys.PN, μ -> scattering_kernel(pn_equ, μ, 1, 1), nd(pn_sys))

    ρ_to_ρp = build_ρ_to_ρp_projection(pn_sys)
    ρ_to_ρm = build_ρ_to_ρm_projection(pn_sys)

    ρs = [project_function(U[2], model, x -> mass_concentrations(pn_equ, x, e)).free_values for e in 1:number_of_elements(pn_equ)]

    temp = assemble_bilinear(∫ρuv, (x -> 1.0, model), U[1], V[1]) # only get the sparsity pattern of ρp and reuse the colptr and rowval vectors
    ρp_colptr = Vector(temp.colptr)
    ρp_rowval = Vector(temp.rowval)
    ρp = [SparseMatrixCSC{Float64, Int64}(n_basis.x.p, n_basis.x.p, ρp_colptr, ρp_rowval, zeros(Float64, length(temp.nzval))) for _ in 1:number_of_elements(pn_equ)]
    ρm = [Diagonal(zeros(Float64, n_basis.x.m)) for _ in 1:number_of_elements(pn_equ)] 
    for e in 1:number_of_elements(pn_equ)
        mul!(ρp[e].nzval, ρ_to_ρp, ρs[e])
        mul!(ρm[e].diag, ρ_to_ρm, ρs[e])
    end

    # odd (m) space basis functions normalization
    return PNSemidiscretization(
        ((n_basis.x.p, n_basis.x.m), (n_basis.Ω.p, n_basis.Ω.m)),
        pn_equ,

        ρp,
        ρm,

        [assemble_bilinear(a, (model, ), U[1], V[1]) for a ∈ ∫absn_uv(model.model)],
        [assemble_bilinear(a, (model, ), U[1], V[2]) for a ∈ ∫∂u_v(model.model)],

        Diagonal(ones(n_basis.Ω.p)),
        # we use negative odd basis function for direction (resulting matrix is symmetric)
        Diagonal(-ones(n_basis.Ω.m)),

        [[Kpp], [Kpp]],
        # we use negative odd basis function for direction (so the resulting matrix is symmetric)
        [[-Kmm], [-Kmm]],

        [Matrix(assemble_boundary_matrix(pn_sys.PN, dir, :pp, nd(model.model))) for dir ∈ space_directions(model.model)],
        [Matrix(assemble_transport_matrix(pn_sys.PN, dir, :pm, nd(model.model))) for dir ∈ space_directions(model.model)],

        # only discretize the even parts (and store as matrices (with 1x or x1 dimension) then CUDA works):
        [sparse(reshape(assemble_linear(∫nzgv, ((x -> beam_position(pn_equ, x, j)), model), U[1], V[1]), (n_basis.x.p, 1),)) for j in 1:number_of_beam_positions(pn_equ)],
        [Matrix(reshape(assemble_direction_boundary(pn_sys.PN, (Ω -> beam_direction(pn_equ, Ω, k)), [0.0, 0.0, 1.0], nd(model.model)).p, (1, n_basis.Ω.p))) for k in 1:number_of_beam_directions(pn_equ)],

        # this is a placeholder for μ+ with ρ = 1
        [Matrix(reshape(assemble_linear(∫μv, (x -> extraction_position(pn_equ, x, e), model), U[1], V[1]), (n_basis.x.p, 1))) for e in 1:number_of_extraction_positions(pn_equ)],
        [Matrix(reshape(assemble_direction_source(pn_sys.PN, (Ω -> extraction_direction(pn_equ, Ω, k)), nd(model.model)).p, (1, n_basis.Ω.p))) for k in 1:number_of_extraction_directions(pn_equ)],

        ρ_to_ρp,
        ρ_to_ρm
    )
end

function equations(pn_semi::PNSemidiscretization)
    return pn_semi.pn_equ
end

function update_mass_concentrations!(pn_semi, ρs)
    for e in 1:number_of_elements(equations(pn_semi))
        mul!(pn_semi.ρp[e].nzval, pn_semi.ρ_to_ρp, ρs[e])
        mul!(pn_semi.ρm[e].diag, pn_semi.ρ_to_ρm, ρs[e])

        # also update the μs
    end
end

# i: energy, j: space, k: direction 
function assemble_beam_rhs!(b, pn_semi::PNSemidiscretization, ϵ, (i, j, k), α)
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size
    np = nLp*nRp
    nm = nLm*nRm

    bp = reshape(@view(b[1:np]), (nLp, nRp))
    bm = reshape(@view(b[np+1:np+nm]), (nLm, nRm))

    mul!(bp, pn_semi.gx[j], pn_semi.gΩ[k], α*beam_energy(pn_semi.pn_equ, ϵ, i), false)
    fill!(bm, 0)
end

function assemble_extraction_rhs!(b, pn_semi::PNSemidiscretization, ϵ, (i, j, k), α)
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size
    np = nLp*nRp
    nm = nLm*nRm

    bp = reshape(@view(b[1:np]), (nLp, nRp))
    bm = reshape(@view(b[np+1:np+nm]), (nLm, nRm))

    mul!(bp, pn_semi.μx[j], pn_semi.μΩ[k], α*extraction_energy(pn_semi.pn_equ, ϵ, i), false)
    fill!(bm, 0)
end

function project_function(U, (model, R, dx, ∂R, dΓ, n), f)
    op = AffineFEOperator((u, v) -> ∫(u*v)dx, v -> ∫(v*f)dx, U, U)
    return Gridap.solve(op)
end


function build_ρ_to_ρp_projection(pn_sys)
    f = project_function(pn_sys.U[2], pn_sys.model, x -> 0.0)
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