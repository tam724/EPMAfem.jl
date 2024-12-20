abstract type AbstractDiscretePNSystem end

"""
    abstract type for excitation/extraction. The co-flag indicates whether the vector is a covector or vector (to be somewhat safe while experimenting).
"""
abstract type AbstractDiscretePNVector{co} end

