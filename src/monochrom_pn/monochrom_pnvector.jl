@concrete struct DiscreteMonochromPNVector
    adjoint::Bool
    model
    arch

    # might be moded to gpu
    bx
    bΩ
end

_is_adjoint_vector(b::DiscreteMonochromPNVector) = b.adjoint

function assemble!(rhs, B::Vector, Δ, sym, β=false)
    for i in eachindex(B)
        assemble!(rhs, B[i], Δ, sym, β)
        β = true
    end
end

function assemble!(rhs, b::DiscreteMonochromPNVector, Δ, sym, β=false)
    rhs_p, rhs_m = pmview(rhs, b.model)
    mul!(rhs_p, b.bx.p, transpose(b.bΩ.p), Δ, β)
    mul!(rhs_m, b.bx.m, transpose(b.bΩ.m), Δ, β)
    # my_rmul!(rhs_m, β) # *-1 if sym
end

function integrate(b::DiscreteMonochromPNVector, ψ)
    ψp, ψm = pmview(ψ, b.model)
    return dot(b.bx.p, ψp, b.bΩ.p) + dot(b.bx.p, ψm, b.bΩ.p)
end
