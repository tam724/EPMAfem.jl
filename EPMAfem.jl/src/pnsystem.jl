@concrete struct DiscretePNSystem <: AbstractDiscretePNSystem
    model

    # energy (these will always live on the cpu)
    s
    τ
    σ

    # space (might be moved to gpu)
    ρp
    ρp_proj
    ρm
    ρm_proj

    ∂p
    ∇pm

    # direction (might be moved to gpu)
    Ip
    Im
    kp
    km
    absΩp
    Ωpm
end

@concrete struct NonAdjointIterator
    system
    rhs
    solver
    reverse
    state
end

function iterator(system::AbstractDiscretePNSystem, rhs::AbstractDiscretePNVector{false}, solver)
    return NonAdjointIterator(system, rhs, solver, false, nothing)
end

function reverse_iterator(system::AbstractDiscretePNSystem, rhs::AbstractDiscretePNVector{false}, solver, state)
    return NonAdjointIterator(system, rhs, solver, true, state)
end

function Base.iterate(it::NonAdjointIterator)
    if it.reverse
        initialize_from_state!(it.solver, it.state)
        ϵs = energy(it.system.model)
        ϵ = ϵs[1]
        return (ϵ, 1), 1
    end

    initialize!(it.solver, it.system)
    ϵs = energy(it.system.model)
    ϵ = ϵs[end]
    return (ϵ, length(ϵs)), length(ϵs)
end

function Base.iterate(it::NonAdjointIterator, i)
    if it.reverse
        ϵs = energy(it.system.model)
        if i >= length(ϵs)
            return nothing
        else
            # here we update the solver state from i to i+1! 
            i = i+1
            ϵi, ϵip1 = ϵs[i-1], ϵs[i]
            Δϵ = ϵip1-ϵi
            step_nonadjoint!(it.solver, it.system, it.rhs, i, -Δϵ)
            return (ϵip1, i), i
        end
    end

    if i <= 1
        return nothing
    else
        ϵs = energy(it.system.model)
        # here we update the solver state from i+1 to i! NOTE: NonAdjoint means from higher to lower energies/times
        ϵi, ϵip1 = ϵs[i-1], ϵs[i]
        Δϵ = ϵip1-ϵi
        step_nonadjoint!(it.solver, it.system, it.rhs, i, Δϵ)
        return (ϵi, i-1), i-1
    end
end

@concrete struct AdjointIterator
    system
    rhs
    solver
end

function iterator(system::AbstractDiscretePNSystem, rhs::AbstractDiscretePNVector{true}, solver)
    return AdjointIterator(system, rhs, solver)
end

function Base.iterate(it::AdjointIterator)
    initialize!(it.solver, it.system)
    ϵs = energy(it.system.model)
    ϵ = ϵs[1]
    return (ϵ, 1), 1
end

function Base.iterate(it::AdjointIterator, i)
    ϵs = energy(it.system.model)
    if i >= length(ϵs)
        return nothing
    else
        # here we update the solver state from i to i+1! NOTE: Adjoint means from lower to higher energies/times
        ϵi, ϵip1 = ϵs[i], ϵs[i+1]
        Δϵ = ϵip1-ϵi
        step_adjoint!(it.solver, it.system, it.rhs, i, Δϵ)
        return (ϵip1, i+1), i+1
    end
end
