using EPMAfem.Krylov

function MinresSolver(m, n, S; window :: Int=5)
    FC = eltype(S)
    T  = real(FC)
    Δx = S(undef, 0)
    x  = S(undef, n)
    r1 = S(undef, n)
    r2 = S(undef, n)
    w1 = S(undef, n)
    w2 = S(undef, n)
    y  = S(undef, n)
    v  = S(undef, 0)
    err_vec = zeros(T, window)
    stats = SimpleStats(0, false, false, T[], T[], T[], 0.0, "unknown")
    solver = MinresSolver{T,FC,S}(m, n, Δx, x, r1, r2, w1, w2, y, v, err_vec, false, stats)
    return solver
  end

function allocate_minres_krylov_buf(VT, m, n; window::Int=5)
    T = eltype(VT)
    Δx = VT(undef, 0)
    # x  = S(undef, n) # stateful
    r1 = VT(undef, n)
    r2 = VT(undef, n)
    w1 = VT(undef, n)
    w2 = VT(undef, n)
    y  = VT(undef, n)
    v  = VT(undef, 0)
    err_vec = zeros(T, window)
    return (Δx, r1, r2, w1, w2, y, v, err_vec)
end

function solver_from_buf(VT, m, n, x, buf)
    T = eltype(VT)
    Δx, r1, r2, w1, w2, y, v, err_vec = buf
    @assert all(sz -> sz == n, (length(x), length(r1), length(r2), length(w1), length(w2), length(y))) 
    stats = SimpleStats(0, false, false, T[], T[], T[], 0.0, "unknown")
    return MinresSolver{T,T,VT}(m, n, Δx, x, r1, r2, w1, w2, y, v, err_vec, false, stats)
end





krylov_buf = EPMAfem.allocate_minres_krylov_buf(Vector{Float64}, 100, 100)

x = zeros(200)
solver = EPMAfem.solver_from_buf(@view(x[1:100]), krylov_buf);

A = rand(100, 100)
A = A*transpose(A)
b = rand(100)

Krylov.solve!(solver, A, b)

A*solver.x.-b

A = rand(10, 10) |> cu
B = rand(10, 10) |> cu
C = rand(10, 10) |> cu

mul!(transpose(A), transpose(B), C, true, false)

function hello(a::Any)
    @show "hello1"
    @show a
    return
end

hello(1.0)
hello((1.0, ))
hello((a=1.0, ))

function hello((a, )::Tuple)
    @show "hello2"
    @show a
    return
end

hello(1.0)
hello((1.0, ))
hello((a=1.0, ))

function hello((;a )::@NamedTuple{a})
    @show "hello3"
    @show a
    return
end

hello(1.0)
hello((1.0, ))
hello((a=1.0, ))

function hello((;a )::NamedTuple{(:a,), Tuple{<:Any}})
    @show "hello3"
    @show a
    return
end

hello(1.0)
hello((1.0, ))
hello((a=1.0, ))

function hello((;a )::NamedTuple{(:a,), <:Tuple{<:Any}})
    @show "hello3"
    @show a
    return
end

hello(1.0)
hello((1.0, ))
hello((a=1.0, ))

function hello2((;a )::NamedTuple{(:a,), Tuple{T}} where T)
    @show "hello4"
    @show a
    return
end

hello(1.0)
hello((1.0, ))
hello((a=1.0, ))

function test((; a, b)::@NamedTuple{a::Float64, b})
    @show a, b
end


function test((; a, b)::@NamedTuple{a::Float64, b::T} where T)
    @show a, b
end

function test((; a, b)::@NamedTuple{a::Int, b})
    @show "ima int"
    @show a, b
end