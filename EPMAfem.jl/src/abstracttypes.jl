# holds the basis matrices for the PNProblem
abstract type AbstractDiscretePNProblem end

# a "solvable" PNProblem, it holds the problem and defines the solver
abstract type AbstractDiscretePNSystem end

# implicit midpoint systems
abstract type AbstractDiscretePNSystemIM <: AbstractDiscretePNSystem end

"""
    abstract type for excitation/extraction. The co-flag indicates whether the vector is a covector or vector (to be somewhat safe while experimenting).
"""
abstract type AbstractDiscretePNVector{co} end

