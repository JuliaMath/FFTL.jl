# Compiling ahead of time

`juliac --trim` (Julia 1.12 and newer) compiles a Julia program into an executable or a library
that contains only the code the program can reach. Every call must be resolvable when the
program is compiled. The normal FFTL plans do not meet this: their type contains the radix
sequence, which is computed from the size at run time, so constructing a plan builds a new type.

For trimmed programs, FFTL has plans whose radix sequences and sizes are data:

| plan | transform | applied with |
|:--|:--|:--|
| `FFTL.RPlan{T}(sz, region, σ)` | complex DFT, sign `σ` | `mul!`, `*`, `\`, or `FFTL.execute!` on pointers |
| `FFTL.RRFFTPlan{T}(sz, region, σ)` | r2c (`σ = -1`) or unnormalised c2r (`σ = +1`) | `FFTL.execute_r2c!`, `FFTL.execute_c2r!` on pointers |
| `FFTL.RR2RPlan{T}(sz, kinds, region)` | real-to-real, one kind per dimension | `FFTL.execute_r2r!` on pointers |
| `FFTL.plan_pass_runtime(v, d)` | batched 1D DFT along `d` of a strided view | `mul!`, `apply!` |
| `FFTL.plan_r2c_pass_runtime(v, d)`, `FFTL.plan_c2r_pass_runtime(v, d)` | batched r2c and c2r along `d` | `mul!` |

`T` is `Float32` or `Float64`; sizes, regions and kinds are vectors. All three take the
`threads` keyword. They compute the same transforms with the same strategies as the normal plans
(resampling, four-step, fused planes); per stage they dispatch at run time, which costs a little
speed. [libfftl](@ref "C library (libfftl)") is built from exactly these plans, so they are the
tested route through `juliac --trim=safe`.

```@example trim
using FFTL, LinearAlgebra

x = randn(ComplexF64, 48, 30)
p = FFTL.RPlan{Float64}([48, 30], [1, 2], -1)
y = similar(x)
mul!(y, p, x)
y ≈ fft(x)
```

```@example trim
r = randn(16)
q = FFTL.RR2RPlan{Float64}([16], [FFTL.REDFT10], [1])
out = similar(r)
GC.@preserve r out FFTL.execute_r2r!(q, pointer(out), pointer(r))
out ≈ FFTL.r2r(r, FFTL.REDFT10)
```

The pass plans of the last two rows are the run-time versions of `plan_pass`, `plan_r2c_pass` and
`plan_c2r_pass`. Their type depends on the element type and the number of dimensions only (for example
`PassPlan{Float64,2,RPassKern{Float64}}`), not on the length, so plans of any size can be stored in one
container of a trimmed program. The specialised plans of the normal functions are faster for short lines.

```@example trim
V = randn(ComplexF64, 64, 5)
pass = FFTL.plan_pass_runtime(V, 1)
W = similar(V)
mul!(W, pass, V)
W ≈ fft(V, 1)
```

The pointer functions expect column-major arrays of the plan's size, which the caller keeps
alive (`GC.@preserve`). The source of libfftl (`libfftl/src/libfftl.jl` and the FFTW3 interface in
`libfftl/src/fftw3api.jl`) shows the rules the calling code follows to stay trimmable: no closures that capture types, concrete element types
at every call, error messages as literal strings.
