using LinearAlgebra
using Plots

function Diagonal(v, m, n)
    D = zeros(m, n)
    D[1:min(m, n), 1:min(m, n)] .+= Diagonal(v)
    return D
end

function tsvd(A, r)
    svd_ = svd(A)
    return LinearAlgebra.SVD(svd_.U[:, 1:r], svd_.S[1:r], svd_.Vt[1:r, :])
end

# computes the rank-(length(M)) update ΔS on S such that diag(transpose(Mᵤ) * U (S + ΔS) * transpose(V) * Mᵥ) ≈ M
function _rank_M_update(S, U, V, Mᵤ, Mᵥ, M)
    A = transpose(Mᵤ) * U      # r × u
    B = transpose(V) * Mᵥ      # v × r

    # Current approximation
    M̂ = diag(A * S * B)
    R = M - M̂

    r, u = size(A)
    v, r2 = size(B)
    @assert r == r2

    C = zeros(r, u * v)

    # Build C matrix row-by-row
    for i in 1:r
        C[i, :] = kron(B[:, i]', A[i, :])  # Row i is A[i,:] ⊗ B[:,i]'
    end

    # Solve least squares: C * vec(ΔS) = R
    vec_ΔS = C \ R
    ΔS = reshape(vec_ΔS, u, v)

    return ΔS
end

n_u = 30
n_v = 20
# starting from the low rank factorization Û Ŝ V̂ᵀ where Û and V̂ contain augmented basis vectors Mᵤ and Mᵥ
# setup Mᵤ, Mᵥ
n_M_u = 2
n_M_v = 2

Mᵤ = qr(rand(n_u, n_M_u)).Q |> Matrix
Mᵥ = qr(rand(n_v, n_M_v)).Q |> Matrix

# setup U₀ and V₀ and augment with Mᵤ and Mᵥ to get Û, V̂
n_S_u = 7; @assert n_S_u - n_M_u > 0
n_S_v = 8; @assert n_S_v - n_M_v > 0

U₀ = qr(rand(n_u, n_S_u - n_M_u)).Q |> Matrix
V₀ = qr(rand(n_v, n_S_v - n_M_v)).Q |> Matrix
Û = qr([Mᵤ U₀]).Q |> Matrix
V̂ = qr([Mᵥ V₀]).Q |> Matrix

# setup Ŝ
Ŝ = rand(n_S_u, n_S_v)
# compute the full matrix (for reference)
Â = Û*Ŝ*transpose(V̂)

# compute the invariants (ideally)
M = diag((transpose(Mᵤ) * Û) * Ŝ * (transpose(V̂) * Mᵥ))

# compute the U₀, V₀ projection
S₀ = (U₀' * Û) * Ŝ * (transpose(V̂) * V₀)
# (for reference)
A₀ = (U₀ * U₀') * Û * Ŝ * transpose(V̂) * (V₀ * V₀')

ΔS₀ = _rank_M_update(S₀, U₀, V₀, Mᵤ, Mᵥ, M)

diag(transpose(Mᵤ) * U₀ * (S₀ + ΔS₀) * transpose(V₀) * Mᵥ)

## 
A = sum(rand(n_u) * rand(n_v)' for i in 1:10)
Mᵥ = zeros(n_v, n_M_v); Mᵥ[1, 1] = 1.0;
Mᵥ[:, 2] = rand(n_v)
Mᵥ = qr(Mᵥ).Q |> Matrix

QR_U = qr(A * Mᵥ)
U_cons = Matrix(QR_U.Q)
σ_cons = Matrix(QR_U.R)

Ã = (I - U_cons*U_cons') * A * (I - Mᵥ*Mᵥ')
Ã = A * (I - Mᵥ*Mᵥ')

svd_Ã = tsvd(Ã, 5)

QR_U = qr([U_cons svd_Ã.U])
QR_V = qr([Mᵥ svd_Ã.V])

S = QR_U.R * [σ_cons zeros(n_M_v, 5); zeros(n_M_v, 5)' Diagonal(svd_Ã.S)] * transpose(QR_V.R)

A_new = Matrix(QR_U.Q) * S * transpose(Matrix(QR_V.Q))

plot(heatmap(A), heatmap(A_new), heatmap(A .- A_new))

plot([A * Mᵥ[:, 1], A_new * Mᵥ[:, 1]])
plot!([A * Mᵥ[:, 2], A_new * Mᵥ[:, 2]])
