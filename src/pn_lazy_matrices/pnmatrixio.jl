const _INDENT = "   "

size_string(A::AbstractMatrix) = "$(size(A, 1))x$(size(A, 2))"

function lazy_print(io::IO, L::LazyMatrix, indent=0)
    println(io, string(repeat(_INDENT, indent)), iszero(indent) ? " " : "∟ ", "($(repr(lazy_objectid(L))))::", size_string(L)," $(typeof(L))")
end

function lazy_print(io::IO, a::AbstractLazyScalar, indent=0)
    println(io, string(repeat(_INDENT, indent)), iszero(indent) ? " " : "∟ ", "($(a[]))::$(typeof(a))")
end

function lazy_print(io::IO, L::LazyOpMatrix, indent=0)
    println(io, string(repeat(_INDENT, indent)), iszero(indent) ? " " : "∟ ", size_string(L), " $(typeof(L)):")
    for A in L.args
        lazy_print(io, A, indent+1)
    end
end

function lazy_print(io::IO, L::Transpose, indent=0)
    println(io, string(repeat(_INDENT, indent)), iszero(indent) ? " " : "∟ ", "$(typeof(L)):")
    lazy_print(io, parent(L), indent+1)
end

function Base.show(io::IO, ::MIME"text/plain", L::LazyOpMatrix) 
    lazy_print(io, L)
    try
        Base.print_matrix(io, L)
    catch
        println(io, "[...]")
    end
end

function Base.show(io::IO, ::MIME"text/plain", L::NotSoLazy)
    println(io, size_string(L), " $(typeof(L)):")
    println(io, "[...]")
end
