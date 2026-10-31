# Complex arithmetic on SIMD vectors holding interleaved complex numbers (re, im, re, im, …).
#
# Vector types used by the kernels:
#   Vec{2,Float64}  one ComplexF64          (128-bit register)
#   Vec{4,Float32}  two ComplexF32          (128-bit register)
#   Vec{2,Float32}  one ComplexF32          (64-bit half register)
#
# On aarch64 with the ARMv8.3 complex-number extension, fcadd90/fcadd270 map to FCADD and
# fcmla0/fcmla90 to FCMLA. Elsewhere the same operations are built from shuffles.

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

# Lane permutations and sign patterns for the shuffle fallback.
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

for (T, N, suf) in ((Float64, 2, "v2f64"), (Float32, 4, "v4f32"), (Float32, 2, "v2f32"))
  VT = NTuple{N,VecElement{T}}
  if HAS_FCMA
    @eval begin
      @inline fcadd90(a::Vec{$N,$T}, b::Vec{$N,$T}) =
        Vec(ccall($("llvm.aarch64.neon.vcadd.rot90." * suf), llvmcall, $VT, ($VT, $VT), a.data, b.data))
      @inline fcadd270(a::Vec{$N,$T}, b::Vec{$N,$T}) =
        Vec(ccall($("llvm.aarch64.neon.vcadd.rot270." * suf), llvmcall, $VT, ($VT, $VT), a.data, b.data))
      @inline fcmla0(c::Vec{$N,$T}, a::Vec{$N,$T}, b::Vec{$N,$T}) =
        Vec(ccall($("llvm.aarch64.neon.vcmla.rot0." * suf), llvmcall, $VT, ($VT, $VT, $VT), c.data, a.data, b.data))
      @inline fcmla90(c::Vec{$N,$T}, a::Vec{$N,$T}, b::Vec{$N,$T}) =
        Vec(ccall($("llvm.aarch64.neon.vcmla.rot90." * suf), llvmcall, $VT, ($VT, $VT, $VT), c.data, a.data, b.data))
    end
  else
    @eval begin
      @inline fcadd90(a::Vec{$N,$T}, b::Vec{$N,$T}) = a + mul_i(b)
      @inline fcadd270(a::Vec{$N,$T}, b::Vec{$N,$T}) = a - mul_i(b)
      @inline fcmla0(c::Vec{$N,$T}, a::Vec{$N,$T}, b::Vec{$N,$T}) = muladd(dup_re(a), b, c)
      @inline fcmla90(c::Vec{$N,$T}, a::Vec{$N,$T}, b::Vec{$N,$T}) = muladd(dup_im(a), mul_i(b), c)
    end
  end
end

# complex product a·w
@inline ccmul(a::Vec{N,T}, w::Vec{N,T}) where {N,T} = fcmla90(fcmla0(zero(Vec{N,T}), a, w), a, w)

# (re, im) repeated over all complex lanes of the register
@generated cconst(::Type{Vec{N,T}}, re, im) where {N,T} =
  :(Vec{$N,$T}(($([isodd(i) ? :($T(re)) : :($T(im)) for i in 1:N]...),)))

# number of complex values per vector
ncomplex(::Type{Vec{N,T}}) where {N,T} = N ÷ 2
