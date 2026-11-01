# Complex arithmetic on SIMD vectors holding interleaved complex numbers (re, im, re, im, …).
#
# Each platform has a ladder of vector types, from the widest register down to one complex value
# per vector (`vec_ladder`): 128-bit registers on aarch64 (and x86 without AVX2), 256-bit with
# AVX2. A kernel stage takes the widest type whose number of complex values divides its stride.
#
# On aarch64 with the ARMv8.3 complex-number extension, a ± i·b maps to FCADD and the complex
# product to two FCMLA. Elsewhere both are written as blends of two results (even lanes from one,
# odd lanes from the other), which x86 compiles to vaddsubpd / vfmaddsubpd.

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

const REG_BYTES = HAS_AVX2 ? 32 : 16

"""
    vec_ladder(T)

Vector types for complex `T` values, widest first, down to one complex value per vector.
"""
vec_ladder(::Type{T}) where {T} =
  Tuple(Vec{b ÷ sizeof(T),T} for b in (REG_BYTES, REG_BYTES ÷ 2, REG_BYTES ÷ 4) if b >= 2sizeof(T))
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
  # a·w = (a.re·w.re - a.im·w.im, a.re·w.im + a.im·w.re)
  @inline function ccmul(a::Vec{N,T}, w::Vec{N,T}) where {N,T}
    t = dup_im(a) * swap_reim(w)
    blend_reim(muladd(dup_re(a), w, -t), muladd(dup_re(a), w, t))
  end
end

# (re, im) repeated over all complex lanes of the register
@generated cconst(::Type{Vec{N,T}}, re, im) where {N,T} =
  :(Vec{$N,$T}(($([isodd(i) ? :($T(re)) : :($T(im)) for i in 1:N]...),)))

# number of complex values per vector
ncomplex(::Type{Vec{N,T}}) where {N,T} = N ÷ 2
