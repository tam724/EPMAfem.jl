@concrete struct HighToLowIterator
    problem
    solver
end

function hightolow(problem::DiscretePNProblem, solver::PNSolver)
    return HighToLowIterator(problem, solver)
end

function Base.iterate(it::HighToLowIterator)
    initialize!(it.solver)
    ϵs = energy(it.problem.model)
    ϵ = ϵs[end]
    return ϵ, length(ϵs)
end

function Base.iterate(it::HighToLowIterator, i)
    if i <= 1
        return nothing
    else
        ϵs = energy(it.problem.model)
        # here we update the solver state from i+1 to i! NOTE: HighToLow means from higher to lower energies/times
        ϵi, ϵip1 = ϵs[i-1], ϵs[i]
        Δϵ = ϵip1-ϵi
        step_hightolow!(it.solver, it.problem, i, Δϵ)
        return ϵi, i-1
    end
end

@concrete struct LowToHighIterator
    problem
    solver
end

function lowtohigh(problem::DiscretePNProblem, solver::PNSolver)
    return LowToHighIterator(problem, solver)
end

function Base.iterate(it::LowToHighIterator)
    initialize!(it.solver)
    ϵs = energy(it.problem.model)
    ϵ = ϵs[1]
    return ϵ, 1
end

function Base.iterate(it::LowToHighIterator, i)
    ϵs = energy(it.problem.model)
    if i >= length(ϵs)
        return nothing
    else
        # here we update the solver state from i to i+1! NOTE: LowToHigh means from lower to higher energies/times
        ϵi, ϵip1 = ϵs[i], ϵs[i+1]
        Δϵ = ϵip1-ϵi
        step_lowtohigh!(it.solver, it.problem, i, Δϵ)
        return ϵip1, i+1
    end
end
