

A = pnproblem.ρp[1]
CUDA.CUSPARSE.CuSparseMatrixCOO(A)

B = CUDA.rand(2099601, 66)
C1 = CUDA.zeros(2099601, 66)
C2 = CUDA.zeros(2099601, 66)

CUDA.@time mul!(C, A, B, 1.0, 0.0)

@benchmark CUDA.@sync CUDA.CUSPARSE.mm!('N', 'N', 1.0, A, B, 0.0, C, 'O')



function run_old!(A, B, C, N)
    for _ in 1:N
        mul!(C, transpose(A), B, 1.0, 0.0)
    end
end

function run_new!(A, B, C, N)
    transa = 'T'
    transb = 'N'
        
    descA = CUDA.CUSPARSE.CuSparseMatrixDescriptor(A, 'O')
    descB = CUDA.CUSPARSE.CuDenseMatrixDescriptor(B)
    descC = CUDA.CUSPARSE.CuDenseMatrixDescriptor(C)

    out = Ref{Csize_t}(10000)
    T = Float32
    algo = CUDA.CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT
    CUDA.CUSPARSE.cusparseSpMM_bufferSize(CUDA.CUSPARSE.handle(), transa, transb, Ref{T}(1.0), descA, descB, Ref{T}(0.0), descC, T, algo, out)
    buffer = CuVector{UInt8}(undef, out[])
    CUDA.CUSPARSE.cusparseSpMM_preprocess(CUDA.CUSPARSE.handle(), transa, transb, Ref{T}(1.0), descA, descB, Ref{T}(0.0), descC, T, algo, buffer)
    #CUDA.CUSPARSE.cusparseSpMM(CUDA.CUSPARSE.handle(), transa, transb, Ref{T}(1.0), descA, descB, Ref{T}(1.0), descC, T, algo, buffer)
    
    for _ in 1:N
        CUDA.CUSPARSE.cusparseSpMM(CUDA.CUSPARSE.handle(), transa, transb, Ref{T}(1.0), descA, descB, Ref{T}(0.0), descC, T, algo, buffer)
    end
end


CUDA.@time run_old!(A, B, C2, 100)

CUDA.@time run_new!(A, B, C1, 100)