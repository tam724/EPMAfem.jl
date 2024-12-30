@concrete struct HighToLowIterator
    problem
    rhs
    solver
end

function hightolow(problem::DiscretePNProblem, rhs::AbstractDiscretePNRHS, solver::PNSolver)
    return HighToLowIterator(problem, rhs, solver)
end

function Base.iterate(it::HighToLowIterator)
    initialize!(it.solver, it.problem)
    ϵs = energy_model(it.problem.model)
    ϵ = ϵs[end]
    return (ϵ, length(ϵs)), length(ϵs)
end

function Base.iterate(it::HighToLowIterator, i)
    if i <= 1
        return nothing
    else
        ϵs = energy_model(it.problem.model)
        # here we update the solver state from i+1 to i! NOTE: HighToLow means from higher to lower energies/times
        ϵi, ϵip1 = ϵs[i-1], ϵs[i]
        Δϵ = ϵip1-ϵi
        step_hightolow!(it.solver, it.problem, it.rhs, i, Δϵ)
        return (ϵi, i-1), i-1
    end
end

@concrete struct LowToHighIterator
    problem
    rhs
    solver
end

function lowtohigh(problem::DiscretePNProblem, rhs::AbstractDiscretePNRHS, solver::PNSolver)
    return LowToHighIterator(problem, rhs, solver)
end

function Base.iterate(it::LowToHighIterator)
    initialize!(it.solver, it.problem)
    ϵs = energy_model(it.problem.model)
    ϵ = ϵs[1]
    return (ϵ, 1), 1
end

function Base.iterate(it::LowToHighIterator, i)
    ϵs = energy_model(it.problem.model)
    if i >= length(ϵs)
        return nothing
    else
        # here we update the solver state from i to i+1! NOTE: LowToHigh means from lower to higher energies/times
        ϵi, ϵip1 = ϵs[i], ϵs[i+1]
        Δϵ = ϵip1-ϵi
        step_lowtohigh!(it.solver, it.problem, it.rhs, i, Δϵ)
        return (ϵip1, i+1), i+1
    end
end
