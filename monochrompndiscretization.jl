

struct DiscreteMonoChromPNProblem{T, V<:AbstractVector{T}, M<:AbstractMatrix{T}, SM<:AbstractSparseMatrix{T}, PNM<:MonoChromPNGridapModel}
    model::PNM

    # energy (these will always live on the cpu)
    μa::Vector{T} # specific absorption cross section
    μs::Matrix{T} # specific total scattering cross section

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

struct DiscreteMonoChromPNRHS{T, V<:AbstractVector{T}, PNM<:MonoChromPNGridapModel}
    model::PNM

    weights::Array{T, 2}

    #(might be moved to gpu)
    bxp::Vector{V}
    bΩp::Vector{V}
end

function discretize_problem(pn_eq::AbstractMonoChromPNEquations, discrete_model::MonoChromPNGridapModel)
    MT = mat_type(discrete_model)
    VT = vec_type(discrete_model)
    SMT = smat_type(discrete_model)
    T = base_type(discrete_model)

    μ_a = T.(specific_absorption_cross_section(pn_eq))
    μ_s = T.(specific_total_scattering_cross_section(pn_eq))

    ## instantiate Gridap stuff
    U, V, gap_model = gridap_model(space(discrete_model))
    n_basis = number_of_basis_functions(discrete_model)

    ## assemble all the space matrices
    ρp = [SMT((assemble_bilinear(∫ρuv, (mass_concentrations(pn_eq, e), gap_model), U[1], V[1]))) for e in 1:number_of_elements(pn_eq)] 
    ρm = [Diagonal(VT(diag(assemble_bilinear(∫ρuv, (mass_concentrations(pn_eq, e), gap_model), U[2], V[2])))) for e in 1:number_of_elements(pn_eq)] 

    ∂p = [SMT(dropzeros(assemble_bilinear(a, (gap_model, ), U[1], V[1]))) for a ∈ ∫absn_uv(space(discrete_model))]
    ∇pm = [SMT(assemble_bilinear(a, (gap_model, ), U[1], V[2])) for a ∈ ∫∂u_v(space(discrete_model))]

    ## assemble all the direction matrices
    Kpp, Kmm = assemble_scattering_matrices(max_degree(discrete_model), specific_scattering_kernel(pn_eq), nd(discrete_model))
    Kpp = Diagonal(VT(Kpp.diag))
    Kmm = Diagonal(VT(-Kmm.diag))
    kp = [[Kpp], [Kpp]]
    # we use negative odd basis function for direction (resulting matrix is symmetric)
    km = [[Kmm], [Kmm]]

    Ip = Diagonal(VT(ones(n_basis.Ω.p)))
    # we use negative odd basis function for direction (resulting matrix is symmetric)
    Im = Diagonal(VT(-ones(n_basis.Ω.m)))

    absΩp = [MT(assemble_boundary_matrix(max_degree(discrete_model), dir, :pp, nd(discrete_model), 1e-4)) for dir ∈ space_directions(discrete_model)]
    Ωpm = [MT(assemble_transport_matrix(max_degree(discrete_model), dir, :pm, nd(discrete_model))) for dir ∈ space_directions(discrete_model)]

    DiscreteMonoChromPNProblem(discrete_model, μ_a, μ_s, ρp, ρm, ∂p, ∇pm, Ip, Im, kp, km, absΩp, Ωpm)
end

function discretize_rhs(pn_eq::AbstractMonoChromPNEquations, discrete_model::MonoChromPNGridapModel)
    VT = vec_type(discrete_model)
    T = base_type(discrete_model)
    U, V, gap_model = gridap_model(space(discrete_model))

    #gxp = VT(assemble_linear(∫nzgv, (excitation_spatial_distribution(pn_eq), gap_model), U[1], V[1]))
    gxp = VT(assemble_linear(∫μv, (excitation_spatial_distribution(pn_eq), gap_model), U[1], V[1]))
    # gΩp = VT(assemble_direction_boundary(max_degree(discrete_model), excitation_direction_distribution(pn_eq), @SVector[0.0, 0.0, 1.0], nd(discrete_model)).p)
    gΩp = VT(assemble_direction_source(max_degree(discrete_model), excitation_direction_distribution(pn_eq), nd(discrete_model)).p)

    return DiscreteMonoChromPNRHS(discrete_model, ones(T, 1, 1), [gxp], [gΩp])
end

function discretize(pn_eq::AbstractMonoChromPNEquations, discrete_model::MonoChromPNGridapModel)
    return discretize_problem(pn_eq, discrete_model), discretize_rhs(pn_eq, discrete_model)
end


function assemble_rhs_p!(b, rhs::DiscreteMonoChromPNRHS; bxp=rhs.bxp, bΩp=rhs.bΩp)
    fill!(b, zero(eltype(b)))

    nLp = length(first(bxp))
    nRp = length(first(bΩp))

    bp = reshape(@view(b[1:nLp*nRp]), (nLp, nRp))
    # bp = pview(b, rhs.model)
    for (xi, bxpi) in zip(1:size(rhs.weights, 1), bxp)
        for (Ωi, bΩpi) in zip(1:size(rhs.weights, 2), bΩp)
            bxpi_mat = reshape(@view(bxpi[:]), (length(bxpi), 1))
            bΩpi_mat = reshape(@view(bΩpi[:]), (1, length(bΩpi)))
            mul!(bp, bxpi_mat, bΩpi_mat, -rhs.weights[xi, Ωi], 1.0)
        end
    end
end

struct MonoChromPNSolver{T, V<:AbstractVector{T}, Tsolv}
    a::Vector{T}
    c::Vector{Vector{T}}
    tmp::V
    tmp2::V
    rhs::V
    lin_solver::Tsolv
end

struct MonoChromSchurPNSolver{T, V<:AbstractVector{T}, Tsolv}
    a::Vector{T}
    c::Vector{Vector{T}}
    tmp::V
    tmp2::V
    tmp3::V
    D::V

    rhs_schur::V
    rhs::V
    sol::V

    lin_solver::Tsolv
end

function pn_monochromsolver(pn_eq::AbstractMonoChromPNEquations, discrete_model::MonoChromPNGridapModel)
    n = number_of_basis_functions(discrete_model)
    VT = vec_type(discrete_model)
    T = base_type(discrete_model)

    n_tot = n.x.p*n.Ω.p + n.x.m*n.Ω.m
    return MonoChromPNSolver(
        Vector{T}(undef, number_of_elements(pn_eq)),
        [Vector{T}(undef, number_of_scatterings(pn_eq)) for _ in 1:number_of_elements(pn_eq)], 
        VT(undef, max(n.x.p, n.x.m)*max(n.Ω.p, n.Ω.m)),
        VT(undef, max(n.Ω.p, n.Ω.m)),
        VT(undef, n_tot),
        Krylov.MinresSolver(n_tot, n_tot, VT)
    )
end

function update_coefficients!(solver::Union{MonoChromPNSolver, MonoChromSchurPNSolver}, problem::DiscreteMonoChromPNProblem)
    for e in 1:length(solver.a)
        solver.a[e] = problem.μa[e]
        for sc in 1:length(solver.c[e])
            solver.a[e] += problem.μs[e][sc]
            solver.c[e][sc] = -problem.μs[e][sc]
        end
    end
    b = 1.0
    return solver.a, b, solver.c
end 

function solve(problem::DiscreteMonoChromPNProblem, rhs::DiscreteMonoChromPNRHS, solver::MonoChromPNSolver{T}) where T
    assemble_rhs_p!(solver.rhs, rhs)
    a, b, c = update_coefficients!(solver, problem)
    A = FullBlockMat(problem.ρp, problem.ρm, problem.∇pm, problem.∂p, problem.Ip, problem.Im, problem.kp, problem.km, problem.Ωpm,  problem.absΩp, a, b, c, solver.tmp, solver.tmp2)
    Krylov.solve!(solver.lin_solver, A, solver.rhs, rtol=T(1e-14), atol=T(1e-14))
end

function pn_monochromschursolver(pn_eq::AbstractMonoChromPNEquations, discrete_model::MonoChromPNGridapModel)
    n = number_of_basis_functions(discrete_model)
    VT = vec_type(discrete_model)
    T = base_type(discrete_model)

    np = n.x.p*n.Ω.p
    n_tot = n.x.p*n.Ω.p + n.x.m*n.Ω.m
    return MonoChromSchurPNSolver(
        Vector{T}(undef, number_of_elements(pn_eq)),
        [Vector{T}(undef, number_of_scatterings(pn_eq)) for _ in 1:number_of_elements(pn_eq)], 
        VT(undef, max(n.x.p, n.x.m)*max(n.Ω.p, n.Ω.m)),
        VT(undef, max(n.Ω.p, n.Ω.m)),
        VT(undef, n.x.m*n.Ω.m),
        VT(undef, n.x.m*n.Ω.m),
        VT(undef, np),
        VT(undef, n_tot),
        VT(undef, n_tot),
        Krylov.MinresSolver(np, np, VT),
    )
end

function _update_D(solver::MonoChromSchurPNSolver{T}, problem::DiscreteMonoChromPNProblem, a, b, c) where T
    # assemble D

    ((_, nLm), (_, nRm)) = problem.model.n_basis
    # tmp_m = @view(pn_solv.tmp[1:nLm*nRm])
    tmp2_m = @view(solver.tmp2[1:nRm])

    fill!(solver.D, zero(T))
    for (ρmz, kmz, az, cz) in zip(problem.ρm, problem.km, a, c)
        tmp2_m .= az*problem.Im.diag
        for (kmzi, czi) in zip(kmz, cz)
            axpy!(czi, kmzi.diag, tmp2_m)
        end

        mul!(reshape(solver.D, (nLm, nRm)), reshape(@view(ρmz.diag[:]), (nLm, 1)), reshape(@view(tmp2_m[:]), (1, nRm)), true, true)
        # axpy!(1.0, tmp_m, pn_solv.D)
    end
end

function _compute_schur_rhs(solver::MonoChromSchurPNSolver, problem::DiscreteMonoChromPNProblem, a, b, c)

    ((nLp, nLm), (nRp, nRm)) = problem.model.n_basis
    
    np = nLp*nRp
    nm = nLm*nRm

    rhsp = reshape(@view(solver.rhs[1:np]), (nLp, nRp))
    rhsm = reshape(@view(solver.rhs[np+1:np+nm]), (nLm, nRm))

    rhs_schurp = reshape(@view(solver.rhs_schur[:]), (nLp, nRp))

    # A_tmp_m = reshape(@view(pn_solv.tmp3[1:nLm*nRm]), (nLm, nRm))

    rhs_schurp .= rhsp
    @view(solver.tmp3[1:nLm*nRm]) .= @view(solver.rhs[np+1:np+nm]) ./ solver.D
    # _mul_mp!(rhs_schurp, pn_solv.A_schur.A, A_tmp_m, -1.0)

    mul!(solver.rhs_schur, DMatrix((transpose(∇pmd) for ∇pmd in problem.∇pm), problem.Ωpm, b, mat_view(solver.tmp, nLp, nRm)), @view(solver.tmp3[1:nLm*nRm]), -1.0, true)
end

function _compute_full_solution_schur(solver::MonoChromSchurPNSolver, problem::DiscreteMonoChromPNProblem, a, b, c)

    ((nLp, nLm), (nRp, nRm)) = problem.model.n_basis

    np = nLp*nRp
    nm = nLm*nRm

    full_p = @view(solver.sol[1:np])
    full_m = @view(solver.sol[np+1:np+nm])
    # full_mm = reshape(full_m, (nLm, nRm))

    # bp = reshape(@view(pn_solv.b[1:np]), (nLp, nRp))
    # bm = reshape(@view(pn_solv.b[np+1:np+nm]), (nLm, nRm))

    full_p .= solver.lin_solver.x

    full_m .= @view(solver.rhs[np+1:np+nm])

    # _mul_pm!(full_mm, pn_solv.A_schur.A, reshape(@view(pn_solv.lin_solver.x[:]), (nLp, nRp)), -1.0)
    mul!(full_m, DMatrix(problem.∇pm, (transpose(Ωpmd) for Ωpmd in problem.Ωpm), b, mat_view(solver.tmp, nLm, nRp)), solver.lin_solver.x, -1.0, true)

    full_m .= full_m ./ solver.D
end



function solve(problem::DiscreteMonoChromPNProblem, rhs::DiscreteMonoChromPNRHS, solver::MonoChromSchurPNSolver{T}) where T
    assemble_rhs_p!(solver.rhs, rhs)
    a, b, c = update_coefficients!(solver, problem)
    _update_D(solver, problem, a, b, c) # assembles the right lower block (diagonal)
    _compute_schur_rhs(solver, problem, a, b, c)
    A_schur = SchurBlockMat(problem.ρp, problem.∇pm, problem.∂p, problem.Ip, problem.kp, problem.Ωpm, problem.absΩp, Diagonal(solver.D), a, b, c, solver.tmp, solver.tmp2, solver.tmp3)
    # A = FullBlockMat(problem.ρp, problem.ρm, problem.∇pm, problem.∂p, problem.Ip, problem.Im, problem.kp, problem.km, problem.Ωpm,  problem.absΩp, a, b, c, solver.tmp, solver.tmp2)
    # Krylov.solve!(solver.lin_solver, A, solver.rhs, rtol=T(1e-14), atol=T(1e-14))
    Krylov.solve!(solver.lin_solver, A_schur, solver.rhs_schur, rtol=T(1e-10), atol=T(1e-10))
    _compute_full_solution_schur(solver, problem, a, b, c)
    return solver.lin_solver.stats
end