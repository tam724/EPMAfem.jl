using LinearAlgebra, BenchmarkTools, Printf
using CUDA

function benchmark_matmul_cpu(n::Integer)
    A = rand(Float64, n, n)
    B = rand(Float64, n, n)
    t = @belapsed $A * $B
    return t
end

function benchmark_matmul_gpu(n::Integer, T::Type{<:Union{Float32,Float64}})
    A_gpu = CUDA.rand(T, n, n)
    B_gpu = CUDA.rand(T, n, n)
    # warm-up, JIT
    C_gpu = A_gpu * B_gpu
    CUDA.synchronize()

    t = @belapsed begin
        C_gpu = $A_gpu * $B_gpu
        CUDA.synchronize()
    end
    return t
end

function run_benchmarks(sizes::Vector{<:Integer})
    println("===========================================================================")
    println(rpad("Size", 8), rpad("CPU (Float64)", 20), rpad("GPU (Float32)", 20), rpad("GPU (Float64)", 20))
    println("---------------------------------------------------------------------------")
    for n in sizes
        # 1) CPU Float64
        cpu_time = benchmark_matmul_cpu(n)
        # 2) GPU Float32
        gpu_fp32_time = benchmark_matmul_gpu(n, Float32)
        # 3) GPU Float64
        gpu_fp64_time = benchmark_matmul_gpu(n, Float64)

        @printf("%-8d  %-18.6e  %-18.6e  %-18.6e\n", n, cpu_time, gpu_fp32_time, gpu_fp64_time)
    end
    println("===========================================================================")
end

sizes = [10, 100, 2^8, 2^10, 2^12, 2^14]
if !CUDA.has_cuda()
    @error "No available CUDA GPU was detected."
    return
end
println("Start testing the performance of matrix multiplication on CPU/GPU:")
run_benchmarks(sizes)

##
using PlotlyLight
xs = sizes .^ 2
# cpu_time = [9.694415e-08, 2.450000e-05, 1.437000e-04,
#     4.696800e-03, 1.689109e-01, 9.036293e+00]
# gpu_fp32_time = [3.760000e-05, 4.460000e-05, 3.950000e-05,
#     1.562000e-04, 5.876900e-03, 3.957396e-01]
# gpu_fp64_time = [4.210000e-05, 6.630000e-05, 1.096000e-04,
#     5.041500e-03, 3.099891e-01, 1.970480e+01]

trace_cpu = Config(x=xs, y=cpu_time, mode="lines+markers", name="CPU (Float64)", line=(width=2, color=:blue))

trace_gpu32 = Config(x=xs, y=gpu_fp32_time, mode="lines+markers", name="GPU (Float32)", line=(width=2, color=:red))

trace_gpu64 = Config(x=xs, y=gpu_fp64_time, mode="lines+markers", name="GPU (Float64)", line=(width=2, color=:orange))

layout = Config(
    title="NVIDIA GeForce RTX 3080",
    xaxis=Config(
        title     = "Data size",
        type      = "log",
        autorange = true,
        tickmode  = "array",
        tickvals  = xs,
    ),
    yaxis=Config(
        title="Time (s)",
        type="log",
        autorange=true,
    ),
    legend=Config(x=0.07, y=0.95),
    margin=Config(l=60, r=20, t=60, b=60),
)
fig = Plot([trace_cpu, trace_gpu32, trace_gpu64], layout)
display(fig)
