# Complex arithmetic on SIMD vectors holding interleaved complex numbers (re, im, re, im, …).
#
# Each platform has a ladder of vector types, from the widest register down to one complex value
# per vector (`vec_ladder`): 128-bit registers on aarch64 (and x86 without AVX2), 256-bit with
# AVX2, 512-bit with AVX-512. A kernel stage takes the widest type whose number of complex values divides its stride.
#
# On aarch64 with the ARMv8.3 complex-number extension, a ± i·b maps to FCADD and the complex
# product to two FCMLA. Elsewhere both are written as blends of two results (even lanes from one,
# odd lanes from the other), which x86 compiles to vaddsubpd / vfmaddsubpd.

using Preferences: @load_preference

"""
    HAS_FCMA

Whether the host CPU has the ARMv8.3 complex-number instructions (FCADD, FCMLA). Apple silicon
always has them; on Linux the `fcma` flag in `/proc/cpuinfo` is checked.
"""
const HAS_FCMA = let
  if Sys.ARCH !== :aarch64
    false
  elseif Sys.isapple()
    true
  elseif Sys.islinux() && isfile("/proc/cpuinfo")
    occursin(r"\bfcma\b", read("/proc/cpuinfo", String))
  else
    false
  end
end

"""
    HAS_AVX2

Whether the host is an x86-64 CPU with AVX2 and FMA (256-bit kernels).
"""
const HAS_AVX2 = Sys.ARCH === :x86_64 &&
  Base.BinaryPlatforms.CPUID.test_cpu_feature(Base.BinaryPlatforms.CPUID.JL_X86_avx2) &&
  Base.BinaryPlatforms.CPUID.test_cpu_feature(Base.BinaryPlatforms.CPUID.JL_X86_fma)

"""
    HAS_AVX512

Whether the host is an x86-64 CPU with AVX-512F (and AVX2/FMA): 512-bit kernels.
"""
const HAS_AVX512 = HAS_AVX2 &&
  Base.BinaryPlatforms.CPUID.test_cpu_feature(Base.BinaryPlatforms.CPUID.JL_X86_avx512f)

"""
    FFTL.REG_BYTES

Width in bytes of the widest vector the CPU kernels use: 64 with AVX-512, 32 with AVX2, 16
otherwise. The `simd_bytes` preference lowers it (for comparisons, or CPUs that reduce the clock
for 512-bit instructions); the next load recompiles FFTL:

```julia
using Preferences, FFTL
Preferences.set_preferences!(FFTL, "simd_bytes" => 32; force = true)
```
"""
const REG_BYTES = min(HAS_AVX512 ? 64 : HAS_AVX2 ? 32 : 16,
                      @load_preference("simd_bytes", 64)::Int)

"""
    vec_ladder(T)

Vector types for complex `T` values, widest first, down to one complex value per vector.
"""
vec_ladder(::Type{T}) where {T} =
  Tuple(Vec{b ÷ sizeof(T),T} for b in (REG_BYTES, REG_BYTES ÷ 2, REG_BYTES ÷ 4, REG_BYTES ÷ 8) if b >= 2sizeof(T))
single_vec(::Type{T}) where {T} = Vec{2,T}

# Lane permutations and sign patterns.
@generated swap_reim(a::Vec{N,T}) where {N,T} =
  :(shufflevector(a, Val($(Tuple(isodd(i) ? i : i - 2 for i in 1:N)))))
@generated dup_re(a::Vec{N,T}) where {N,T} =
  :(shufflevector(a, Val($(Tuple(isodd(i) ? i - 1 : i - 2 for i in 1:N)))))
@generated dup_im(a::Vec{N,T}) where {N,T} =
  :(shufflevector(a, Val($(Tuple(isodd(i) ? i : i - 1 for i in 1:N)))))
@generated sign_re(::Type{Vec{N,T}}) where {N,T} =
  :(Vec{$N,$T}($(Tuple(isodd(i) ? -one(T) : one(T) for i in 1:N))))

# i·b = (-b.im, b.re)
@inline mul_i(b::Vec{N,T}) where {N,T} = swap_reim(b) * sign_re(Vec{N,T})
# even lanes (real parts) from x, odd lanes (imaginary parts) from y
@generated blend_reim(x::Vec{N,T}, y::Vec{N,T}) where {N,T} =
  :(shufflevector(x, y, Val($(Tuple(isodd(i) ? i - 1 : N + i - 1 for i in 1:N)))))

if HAS_FCMA
  for (T, N, suf) in ((Float64, 2, "v2f64"), (Float32, 4, "v4f32"), (Float32, 2, "v2f32"))
    VT = NTuple{N,VecElement{T}}
    @eval begin
      @inline fcadd90(a::Vec{$N,$T}, b::Vec{$N,$T}) =
        Vec(ccall($("llvm.aarch64.neon.vcadd.rot90." * suf), llvmcall, $VT, ($VT, $VT), a.data, b.data))
      @inline fcadd270(a::Vec{$N,$T}, b::Vec{$N,$T}) =
        Vec(ccall($("llvm.aarch64.neon.vcadd.rot270." * suf), llvmcall, $VT, ($VT, $VT), a.data, b.data))
      @inline fcmla0(c::Vec{$N,$T}, a::Vec{$N,$T}, b::Vec{$N,$T}) =
        Vec(ccall($("llvm.aarch64.neon.vcmla.rot0." * suf), llvmcall, $VT, ($VT, $VT, $VT), c.data, a.data, b.data))
      @inline fcmla90(c::Vec{$N,$T}, a::Vec{$N,$T}, b::Vec{$N,$T}) =
        Vec(ccall($("llvm.aarch64.neon.vcmla.rot90." * suf), llvmcall, $VT, ($VT, $VT, $VT), c.data, a.data, b.data))
      @inline ccmul(a::Vec{$N,$T}, w::Vec{$N,$T}) = fcmla90(fcmla0(zero(Vec{$N,$T}), a, w), a, w)
      @inline ccmul_at(a::Vec{$N,$T}, p::Ptr{$T}) = ccmul(a, cbroadcast(Vec{$N,$T}, p))
    end
  end
else
  # a + i·b = (a.re - b.im, a.im + b.re)
  @inline fcadd90(a::Vec{N,T}, b::Vec{N,T}) where {N,T} = (sb = swap_reim(b); blend_reim(a - sb, a + sb))
  # a - i·b = (a.re + b.im, a.im - b.re)
  @inline fcadd270(a::Vec{N,T}, b::Vec{N,T}) where {N,T} = (sb = swap_reim(b); blend_reim(a + sb, a - sb))
  if HAS_AVX2
    # x86 addsub(a, c) = (a0 - c0, a1 + c1, …): a + i·b = addsub(a, swap(b)), a - i·b = addsub(a, -swap(b))
    for (T, N, name) in ((Float64, 4, "llvm.x86.avx.addsub.pd.256"), (Float32, 8, "llvm.x86.avx.addsub.ps.256"),
                         (Float64, 2, "llvm.x86.sse3.addsub.pd"), (Float32, 4, "llvm.x86.sse3.addsub.ps"))
      VT = NTuple{N,VecElement{T}}
      @eval begin
        @inline addsub(a::Vec{$N,$T}, c::Vec{$N,$T}) = Vec(ccall($name, llvmcall, $VT, ($VT, $VT), a.data, c.data))
        @inline fcadd90(a::Vec{$N,$T}, b::Vec{$N,$T}) = addsub(a, swap_reim(b))
        @inline fcadd270(a::Vec{$N,$T}, b::Vec{$N,$T}) = addsub(a, -swap_reim(b))
      end
    end
  end
  if HAS_AVX512
    # no 512-bit addsub: fmaddsub(a, 1, c) = (a0 - c0, a1 + c1, …), exact (instead of sub, add, blend)
    for (T, N, name) in ((Float64, 8, "llvm.x86.avx512.vfmaddsub.pd.512"), (Float32, 16, "llvm.x86.avx512.vfmaddsub.ps.512"))
      VT = NTuple{N,VecElement{T}}
      @eval begin
        @inline addsub(a::Vec{$N,$T}, c::Vec{$N,$T}) =
          Vec(ccall($name, llvmcall, $VT, ($VT, $VT, $VT, Int32), a.data, one(Vec{$N,$T}).data, c.data, Int32(4)))
        @inline fcadd90(a::Vec{$N,$T}, b::Vec{$N,$T}) = addsub(a, swap_reim(b))
        @inline fcadd270(a::Vec{$N,$T}, b::Vec{$N,$T}) = addsub(a, -swap_reim(b))
      end
    end
  end
  # a·w = (a.re·w.re - a.im·w.im, a.im·w.re + a.re·w.im) from re(w) and im(w), each repeated over
  # the two lanes of its complex value
  @inline function ccmul_parts(a::Vec{N,T}, wre::Vec{N,T}, wim::Vec{N,T}) where {N,T}
    t = swap_reim(a) * wim
    blend_reim(muladd(a, wre, -t), muladd(a, wre, t))
  end
  @inline ccmul(a::Vec{N,T}, w::Vec{N,T}) where {N,T} = ccmul_parts(a, dup_re(w), dup_im(w))
  # a·w for the complex value w at p, repeated over all complex lanes: re(w) and im(w) are scalar
  # broadcasts, which x86 folds into the multiplies as memory operands
  @inline ccmul_at(a::Vec{N,T}, p::Ptr{T}) where {N,T} =
    ccmul_parts(a, Vec{N,T}(unsafe_load(p)), Vec{N,T}(unsafe_load(p, 2)))
end

# (re, im) repeated over all complex lanes of the register
@generated cconst(::Type{Vec{N,T}}, re, im) where {N,T} =
  :(Vec{$N,$T}(($([isodd(i) ? :($T(re)) : :($T(im)) for i in 1:N]...),)))

# the complex value at p, repeated over all complex lanes (one load and a broadcast)
@generated function cbroadcast(::Type{Vec{N,T}}, p::Ptr{T}) where {N,T}
  :(shufflevector(vload(Vec{2,T}, p), Val($(Tuple(isodd(i) ? 0 : 1 for i in 1:N)))))
end

# complex conjugate of every complex lane
@inline cconj(v::Vec{N,T}) where {N,T} = v * conj_signs(Vec{N,T})
@generated conj_signs(::Type{Vec{N,T}}) where {N,T} = :(Vec{$N,$T}($(Tuple(isodd(i) ? one(T) : -one(T) for i in 1:N))))

# number of complex values per vector
ncomplex(::Type{Vec{N,T}}) where {N,T} = N ÷ 2
