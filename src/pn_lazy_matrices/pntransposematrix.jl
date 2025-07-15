const TransposeMatrix{T} = LazyOpMatrix{T, typeof(transpose), <:Tuple{<:AbstractMatrix{T}}}
A(At::TransposeMatrix) = only(At.args)
Base.size(At::TransposeMatrix) = reverse(size(A(At)))
max_size(At::TransposeMatrix) = reverse(max_size(A(At)))
Base.getindex(At::TransposeMatrix, idx::Vararg{<:Integer}) = getindex(A(At), reverse(idx)...)

## don't know if this makes sense, well see.. (there is Transpose{T, <:AbstractMatrix{T}} already in Base, we can simply use it..)
## the specializations will look similar anyways (might have an impact if there is at some point an actual performance measuring and optimization, then we have our own type)
