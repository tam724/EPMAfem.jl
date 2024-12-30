using EPMAfem.ConcreteStructs
using LinearAlgebra

@concrete struct MyStruct
    mydata
end

function discretize()
    x = rand(10)
    return [MyStruct(x) for i in 1:5]
end

vec = discretize()

function only_unique(arr::Array{<:MyStruct})
    

function (arr::Array{<:MyStruct})(b::Vector)
    ret = zeros(size(arr))
    for i in eachindex(arr)
        ret[i] = dot(arr[i].mydata, b)
    end
    return ret
end

b = rand(10)
vec(b)


function blow_up(A, B)
    arr = zeros((size(A)..., size(B)...))
    for i in eachindex(A)
        arr[i, axes(B)...] = B
    end
