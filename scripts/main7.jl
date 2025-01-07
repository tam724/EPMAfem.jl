using Revise

using EPMAfem
using EPMAfem.CUDA
SH =  EPMAfem.SphericalHarmonicsModels
using Plots
using LinearAlgebra
using LinearOperators
using EPMAfem.Krylov
using LinearMaps

N = 20
direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(N, 3)
basis_harmonics = SH.get_basis_harmonics(direction_model)
SH.degree.(basis_harmonics)

sort!(basis_harmonics, lt=(m1, m2) -> SH.degree(m1) < SH.degree(m2))
SH.degree.(basis_harmonics)

idx = SH.index_findonly.(N, basis_harmonics)
n_by_degree = [length(findall(m -> SH.degree(m) == i, basis_harmonics)) for i in 0:N]
n_deg = cumsum(n_by_degree)
deg_ranges = [(i==1 ? 1 : n_deg[i-1]+1):n_deg[i] for i in 1:length(n_deg)]
Ωpm = [SH.assemble_bilinear(∫, direction_model, idx, idx, SH.exact_quadrature()) for ∫ ∈ SH.∫S²_Ωuv(SH.dimensionality(direction_model))]
absΩp = [SH.assemble_bilinear(∫, direction_model, idx, idx, SH.lebedev_quadrature_max()) for ∫ ∈ SH.∫S²_absΩuv(SH.dimensionality(direction_model))]
for absΩpi in absΩp
    for i in 1:length(idx), j in 1:length(idx)
        if !SH.is_even(basis_harmonics[i]) || !SH.is_even(basis_harmonics[j])
            absΩpi[i, j] = 0.0
        end
    end
end
mat = Ωpm[1] + Ωpm[2] + Ωpm[3] + I
# for i in 1:size(mat, 1)
#     for j in (i+1):size(mat, 2)
#         mat[j, i] *= -1
#     end
# end
mat
# solve this system with iterated schur complements

# function solve_schur(deg, BTAm1B, b_aug, deg_ranges, mat, x, b)
#     b_i = @view(b[deg_ranges[deg]])
#     x_i = @view(x[deg_ranges[deg]])
#     solver_i = MinresSolver(length(deg_ranges[deg]), length(deg_ranges[deg]), Vector{Float64})
#     if deg == 1
#         # A_i_inv = InverseMap(@view(mat[deg_ranges[deg], deg_ranges[deg]]); solver=(x, A, b) -> begin stats = minres!(solver_i, A, b); @show stats; copyto!(x, solver_i.x);end)
#         A_i_inv = inv(@view(mat[deg_ranges[deg], deg_ranges[deg]]))
#         @show deg, cond(@view(mat[deg_ranges[deg], deg_ranges[deg]]))
#     else
#         # A_i_inv = InverseMap(@view(mat[deg_ranges[deg], deg_ranges[deg]]) - BTAm1B; solver=(x, A, b) -> begin stats = minres!(solver_i, A, b); @show stats; copyto!(x, solver_i.x);end)
#         A_i_inv = inv(@view(mat[deg_ranges[deg], deg_ranges[deg]]) - BTAm1B)
#         @show deg, cond(@view(mat[deg_ranges[deg], deg_ranges[deg]]) - BTAm1B)
#     end
#     if deg == N+1
#         x_i .= A_i_inv * b_aug
#     else
#         B_i = @view(mat[deg_ranges[deg], deg_ranges[deg+1]])
#         BTAm1B_next = transpose(B_i)*A_i_inv*B_i
#         next_b_aug = @view(b[deg_ranges[deg+1]]) - transpose(B_i)*A_i_inv*b_aug
#         solve_schur(deg + 1, BTAm1B_next, next_b_aug, deg_ranges, mat, x, b)
#         x_next = @view(x[deg_ranges[deg+1]])
#         x_i .= A_i_inv * (b_aug - B_i*x_next)
#     end
# end

# b = rand(size(mat, 1))
# x = zeros(size(mat, 2))
# solve_schur(1, nothing, @view(b[1:1]), deg_ranges, mat, x, b)

function solve_schur_high_to_low(deg, BTAp1B, b_aug, deg_ranges, mat, x, b)
    # @show "call schur solver for degree", deg, deg_ranges[deg]
    b_i = @view(b[deg_ranges[deg]])
    x_i = @view(x[deg_ranges[deg]])
    solver_i = MinresSolver(length(deg_ranges[deg]), length(deg_ranges[deg]), Vector{Float64})
    A_i = mat[deg_ranges[deg], deg_ranges[deg]]
    if deg != N+1
        A_i .-= BTAp1B
    end
    A_i_inv = inv(A_i)
    # @show deg, cond(A_i)

    # if deg == N+1
    #     # A_i_inv = InverseMap(@view(mat[deg_ranges[deg], deg_ranges[deg]]); solver=(x, A, b) -> begin stats = minres!(solver_i, A, b); @show stats; copyto!(x, solver_i.x);end)
    #     A_i_inv = inv(@view(mat[deg_ranges[deg], deg_ranges[deg]]))
    #     @show deg, cond(@view(mat[deg_ranges[deg], deg_ranges[deg]]))
    # else
    #     # A_i_inv = InverseMap(@view(mat[deg_ranges[deg], deg_ranges[deg]]) - BTAm1B; solver=(x, A, b) -> begin stats = minres!(solver_i, A, b); @show stats; copyto!(x, solver_i.x);end)
    #     A_i_inv = inv(@view(mat[deg_ranges[deg], deg_ranges[deg]]) - BTAp1B)
    #     @show deg, cond(@view(mat[deg_ranges[deg], deg_ranges[deg]]) - BTAp1B)
    # end
    if deg == 1
        x_i .= A_i_inv * b_aug
        # @show deg, b_aug, x_i
    else
        BT_i = @view(mat[deg_ranges[deg], deg_ranges[deg-1]])
        # @show BT_i
        BABT = transpose(BT_i)*A_i_inv*BT_i
        next_b_aug = @view(b[deg_ranges[deg-1]]) - transpose(BT_i)*A_i_inv*b_aug
        solve_schur_high_to_low(deg - 1, BABT, next_b_aug, deg_ranges, mat, x, b)
        x_next = @view(x[deg_ranges[deg-1]])
        x_i .= A_i_inv * (b_aug - BT_i*x_next)
        # @show deg, b_aug, x_i
    end
    return x
end

using BenchmarkTools
b = rand(size(mat, 1))
x = zeros(size(mat, 2))
@benchmark solve_schur_high_to_low(N+1, nothing, @view(b[deg_ranges[end]]), deg_ranges, mat, x, b)


mat*x.-b
mat = sparse(mat)
@benchmark x .= mat\b
@benchmark x .= mat\b
x
    
A = discrete_problem.ρp[1] .* Float32(1e6)

y = zeros(101*201, 200)|> cu
x = rand(101*201, 200)|> cu

function f1!(y, A, x)
    mul!(y, A, x, true, false)
    CUDA.synchronize()
end
function f2!(y, A, x)
    n = size(y, 2)
    mul!(@view(y[:, 1:n÷2]), A, @view(x[:, 1:n÷2]), true, false)
    mul!(@view(y[:, (n÷2+1):n]), A, @view(x[:, (n÷2+1):n]), true, false)
    CUDA.synchronize()
    return y
end
using BenchmarkTools
@benchmark f1!(y, A, x)
@benchmark f2!(y, A, x)

using EPMAfem.Krylov

b = randn(20301) |> cu

x, stats = minres(A, b, atol=Float32(0.0), rtol=Float32(sqrt(eps(Float64))))


A*x .- b