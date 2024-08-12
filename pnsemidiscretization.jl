
using StaticArrays

"""
    DiscretePNProblem

    discrete version of [`PNEquations`](@ref)
"""
struct DiscretePNProblem{T, V<:AbstractVector{T}, M<:AbstractMatrix{T}, SM<:AbstractSparseMatrix{T}, PNM}
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
    Ωpm::Vector{SM}

    # # excitation (might be moved to gpu)
    # gϵ::Vector{T}
    # gxp::M
    # gΩp::M

    # # extraction (might be moved to gpu)
    # μϵ::Vector{T}
    # μxp::M
    # μΩp::M
end

struct DiscretePNRHS{T, M<:AbstractMatrix{}, PNM}
    model::PNM

    # excitation (might be moved to gpu)
    weights::Array{T, 3}

    bϵ::Vector{Vector{T}}
    bxp::Vector{M}
    bΩp::Vector{M}
end

mat_type(::DiscretePNProblem{T, V, M}) where {T, V, M} = M
vec_type(::DiscretePNProblem{T, V, M}) where {T, V, M} = V
base_type(::DiscretePNProblem{T, V, M}) where {T, V, M} = T

function discretize_problem(pn_eq, discrete_model)
    MT = mat_type(discrete_model)
    VT = vec_type(discrete_model)
    SMT = smat_type(discrete_model)
    T = base_type(discrete_model)
    ## assemble (compute) all the energy matrices
    s = Matrix{T}([_s(pn_eq, e)(ϵ) for e in 1:number_of_elements(pn_eq), ϵ ∈ energy(discrete_model)])
    τ = Matrix{T}([_τ(pn_eq, e)(ϵ) for e in 1:number_of_elements(pn_eq), ϵ ∈ energy(discrete_model)])
    σ = Array{T}([_σ(pn_eq, e, i)(ϵ) for e in 1:number_of_elements(pn_eq), i in 1:number_of_scatterings(pn_eq), ϵ ∈ energy(discrete_model)])

    ## instantiate Gridap stuff
    space_model = space(discrete_model)
    U, V = function_spaces(space_model)
    R = Triangulation(space_model)
    ∂R = BoundaryTriangulation(space_model)
    model = model=(model=space_model, R=R, dx=Measure(R, 2), ∂R=∂R, dΓ= Measure(∂R, 2), n=get_normal_vector(∂R))

    n_basis = number_of_basis_functions(discrete_model)

    ## assemble all the space matrices
    ρp = [SMT((assemble_bilinear(∫ρuv, (_mass_concentrations(pn_eq, e), model), U[1], V[1]))) for e in 1:number_of_elements(pn_eq)] 
    ρm = [Diagonal(VT(diag(assemble_bilinear(∫ρuv, (_mass_concentrations(pn_eq, e), model), U[2], V[2])))) for e in 1:number_of_elements(pn_eq)] 

    ∂p = [SMT(dropzeros(assemble_bilinear(a, (model, ), U[1], V[1]))) for a ∈ ∫absn_uv(space(discrete_model))]
    ∇pm = [SMT(assemble_bilinear(a, (model, ), U[1], V[2])) for a ∈ ∫∂u_v(space(discrete_model))]

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

    absΩp = [MT(assemble_boundary_matrix(max_degree(discrete_model), dir, :pp, nd(discrete_model), 1e-7)) for dir ∈ space_directions(discrete_model)]
    Ωpm = [SMT(assemble_transport_matrix(max_degree(discrete_model), dir, :pm, nd(discrete_model))) for dir ∈ space_directions(discrete_model)]

    DiscretePNProblem(discrete_model, s, τ, σ, ρp, ρm, ∂p, ∇pm, Ip, Im, kp, km, absΩp, Ωpm)
end

function discretize_rhs(pn_eq, discrete_model)
    MT = mat_type(discrete_model)
    T = base_type(discrete_model)

    ## instantiate Gridap stuff
    space_model = space(discrete_model)
    U, V = function_spaces(space_model)
    R = Triangulation(space_model)
    ∂R = BoundaryTriangulation(space_model)
    model = model=(model=space_model, R=R, dx=Measure(R, 2), ∂R=∂R, dΓ= Measure(∂R, 2), n=get_normal_vector(∂R))

    n_basis = number_of_basis_functions(discrete_model)

    ## assemble excitation 
    gϵ = Vector{T}([_excitation_energy_distribution(pn_eq)(ϵ) for ϵ ∈ energy(discrete_model)])
    gxp = MT(reshape(assemble_linear(∫nzgv, (_excitation_spatial_distribution(pn_eq), model), U[1], V[1]), (n_basis.x.p, 1),))
    gΩp = MT(reshape(assemble_direction_boundary(max_degree(discrete_model), _excitation_direction_distribution(pn_eq), @SVector[0.0, 0.0, 1.0], nd(discrete_model)).p, (1, n_basis.Ω.p)))

    return DiscretePNRHS(discrete_model, ones(T, 1, 1, 1), [gϵ], [gxp], [gΩp])
end

function discretize_extraction(pn_eq, discrete_model)
    MT = mat_type(discrete_model)
    T = base_type(discrete_model)

    ## instantiate Gridap stuff
    space_model = space(discrete_model)
    U, V = function_spaces(space_model)
    R = Triangulation(space_model)
    ∂R = BoundaryTriangulation(space_model)
    model = model=(model=space_model, R=R, dx=Measure(R, 2), ∂R=∂R, dΓ= Measure(∂R, 2), n=get_normal_vector(∂R))

    n_basis = number_of_basis_functions(discrete_model)

    ## ... and extraction
    μϵ = Vector{T}([_extraction_energy_distribution(pn_eq)(ϵ) for ϵ ∈ energy(discrete_model)])
    μxp = MT(reshape(assemble_linear(∫μv, (_extraction_spatial_distribution(pn_eq), model), U[1], V[1]), (n_basis.x.p, 1)))
    μΩp = MT(reshape(assemble_direction_source(max_degree(discrete_model), _extraction_direction_distribution(pn_eq), nd(discrete_model)).p, (1, n_basis.Ω.p)))

    return DiscretePNRHS(discrete_model, ones(T, 1, 1, 1), [μϵ], [μxp], [μΩp])
end

function discretize(pn_eq, discrete_model)
    return discretize_problem(pn_eq, discrete_model), discretize_rhs(pn_eq, discrete_model)
end

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
    Ωpm::Vector{SM}

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

function pn_semidiscretization(pn_sys, pn_equ, skip_projections=false)
    U = pn_sys.U    
    V = pn_sys.V
    model = pn_sys.model

    n_basis = number_of_basis_functions(pn_sys)
    # TODO!!! (support multiple scattering kernels per element)
    Kpp, Kmm = assemble_scattering_matrices(pn_sys.PN, _electron_scattering_kernel(pn_equ, 1, 1), nd(pn_sys))

    ρ_to_ρp = sparse(zeros(1, 1))
    ρ_to_ρm = Diagonal(zeros(1))
    ρp = nothing
    ρm = nothing
    if !skip_projections
        ρ_to_ρp = build_ρ_to_ρp_projection(pn_sys)
        ρ_to_ρm = build_ρ_to_ρm_projection(pn_sys)

        ρs = [project_function(U[2], model, _mass_concentrations(pn_equ, e)).free_values for e in 1:number_of_elements(pn_equ)]

        temp = assemble_bilinear(∫ρuv, (x -> 1.0, model), U[1], V[1]) # only get the sparsity pattern of ρp and reuse the colptr and rowval vectors
        ρp_colptr = Vector(temp.colptr)
        ρp_rowval = Vector(temp.rowval)
        ρp = [SparseMatrixCSC{Float64, Int64}(n_basis.x.p, n_basis.x.p, ρp_colptr, ρp_rowval, zeros(Float64, length(temp.nzval))) for _ in 1:number_of_elements(pn_equ)]
        ρm = [Diagonal(zeros(Float64, n_basis.x.m)) for _ in 1:number_of_elements(pn_equ)] 
        for e in 1:number_of_elements(pn_equ)
            mul!(ρp[e].nzval, ρ_to_ρp, ρs[e])
            mul!(ρm[e].diag, ρ_to_ρm, ρs[e])
        end
    else
        ρp = [assemble_bilinear(∫ρuv, (_mass_concentrations(pn_equ, e), model), U[1], V[1]) for e in 1:number_of_elements(pn_equ)]
        ρm = [Diagonal(Vector(diag(assemble_bilinear(∫ρuv, (_mass_concentrations(pn_equ, e), model), U[2], V[2])))) for e in 1:number_of_elements(pn_equ)] 
    end
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

        [assemble_boundary_matrix(pn_sys.PN, dir, :pp, nd(model.model)) for dir ∈ space_directions(model.model)],
        [assemble_transport_matrix(pn_sys.PN, dir, :pm, nd(model.model)) for dir ∈ space_directions(model.model)],

        # only discretize the even parts (and store as matrices (with 1x or x1 dimension) then CUDA works):
        [sparse(reshape(assemble_linear(∫nzgv, (_excitation_spatial_distribution(pn_equ, j), model), U[1], V[1]), (n_basis.x.p, 1),)) for j in 1:number_of_beam_positions(pn_equ)],
        [Matrix(reshape(assemble_direction_boundary(pn_sys.PN, _excitation_direction_distribution(pn_equ, k), @SVector[0.0, 0.0, 1.0], nd(model.model)).p, (1, n_basis.Ω.p))) for k in 1:number_of_beam_directions(pn_equ)],

        [Matrix(reshape(assemble_linear(∫μv, (_extraction_spatial_distribution(pn_equ, e), model), U[1], V[1]), (n_basis.x.p, 1))) for e in 1:number_of_extraction_positions(pn_equ)],
        [Matrix(reshape(assemble_direction_source(pn_sys.PN, _extraction_direction_distribution(pn_equ, k), nd(model.model)).p, (1, n_basis.Ω.p))) for k in 1:number_of_extraction_directions(pn_equ)],

        ρ_to_ρp,
        ρ_to_ρm
    )
end

function equations(pn_semi::PNSemidiscretization)
    return pn_semi.pn_equ
end

mat_type(pn_semi::PNSemidiscretization{T, V, M}) where {T, V, M} = M
vec_type(pn_semi::PNSemidiscretization{T, V, M}) where {T, V, M} = V
base_type(pn_semi::PNSemidiscretization{T, V, M}) where {T, V, M} = T

function update_mass_concentrations!(pn_semi, ρs)
    for e in 1:number_of_elements(equations(pn_semi))
        mul!(pn_semi.ρp[e].nzval, pn_semi.ρ_to_ρp, ρs[e])
        mul!(pn_semi.ρm[e].diag, pn_semi.ρ_to_ρm, ρs[e])

        # also update the μs
    end
end

function assemble_rhs_p!(b, rhs, i, Δ; bxp=rhs.bxp, bΩp=rhs.bΩp)
    fill!(b, zero(eltype(b)))

    nLp = size(first(bxp), 1)
    nRp = size(first(bΩp), 2)

    bp = reshape(@view(b[1:nLp*nRp]), (nLp, nRp))
    # bp = pview(b, rhs.model)
    for (ϵi, bϵi) in zip(1:size(rhs.weights, 1), rhs.bϵ)
        for (xi, bxpi) in zip(1:size(rhs.weights, 2), bxp)
            for (Ωi, bΩpi) in zip(1:size(rhs.weights, 3), bΩp)
                bϵ2 = 0.5*(bϵi[i] + bϵi[i+1])
                mul!(bp, bxpi, bΩpi, rhs.weights[ϵi, xi, Ωi]*bϵ2*Δ, 1.0)
            end
        end
    end
end

function assemble_rhs_hightolow!(b, problem::DiscretePNProblem, gϵ, α)
    bp = pview(b, problem.model)
    bm = mview(b, problem.model)

    mul!(bp, problem.gx, problem.gΩ, α*gϵ, false)
    fill!(bm, zero(eltype(b)))
end

function assemble_extraction_rhs!(b, pn_semi::PNSemidiscretization, ϵ, (i, j, k), α)
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size
    np = nLp*nRp
    nm = nLm*nRm

    bp = reshape(@view(b[1:np]), (nLp, nRp))
    bm = reshape(@view(b[np+1:np+nm]), (nLm, nRm))

    mul!(bp, pn_semi.μx[j], pn_semi.μΩ[k], α*_extraction_energy_distribution(pn_semi.pn_equ, i)(ϵ), false)
    fill!(bm, 0)
end

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