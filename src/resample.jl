"""
    Resample{T}

DFT of any length n by oversampled interpolation (a type-2 NUFFT at the points k·m/n): the input,
divided by the Fourier transform of the kernel, is zero-padded to a smooth length m ≥ 2n and
transformed; output k is interpolated from the w grid values around k·m/n with the
exponential-of-semicircle kernel ψ(z) = exp(β(√(1-z²)-1)). One transform of length 2n where
Bluestein's algorithm takes two; the error is about exp(-πw√(1-n/m)) relative to ‖x‖.
"""
struct Resample{T,F,WS} <: Kernel{T}
  n::Int
  corr::Vector{T}                 # 1/Φ̂(j'/m) for input j (j' = j, or j - n for the upper half)
  ws::WS                          # interpolation weights (RTable or RPoly)
  fwd::F                          # length m, sign σ
  buf::Vector{Complex{T}}
  work1::Vector{Complex{T}}
  work2::Vector{Complex{T}}
end

# Output k reads grid values l, …, l+w-1 (mod m) with l = q - h + 1, k·m = q·n + r; its weights
# depend only on δ = r/n. Either a table of the weights of every output, each stored twice for
# the re and im lanes of the interleaved grid (no shuffles to duplicate them), or P polynomial
# pieces of degree d in δ (Horner across the taps, then one duplication), coefficient c of tap i
# on piece p at C[(p(d+1) + c)w + i + 1]. Both are concrete per T (the run-time plans hold them).
struct RTable{T}
  wt::Vector{T}
  l0::Vector{Int32}
end
struct RPoly{T}
  C::Vector{T}
  P::Int
  d::Int
end

resample_tol(::Type{T}) where {T} = 10 * Float64(eps(T))

# Narrowest multiple of 4 meeting the tolerance at m ≥ 2n; a constant per T (16 for Float64, 8 for
# Float32), so the kernels take it from the element type.
resample_width(::Type{T}) where {T} = 4 * cld(ceil(Int, log(1 / resample_tol(T)) / (π * sqrt(0.5))), 4)

# Largest weight table; above it the polynomial weights (no table streamed through the cache). At
# parity on Tiger Lake: F64 near 1 MiB, F32 (8 taps, less Horner work per byte saved) near 4 MiB.
resample_table_bytes(::Type{T}) where {T} = sizeof(T) >= 8 ? 1 << 20 : 1 << 22

# C: TypedKernels, or RChain for run-time plans (their grid transforms are chains)
function Resample{T}(n::Int, σ::Int, ::Type{C} = TypedKernels) where {T,C}
  m, _, corr, ws = resample_parts(T, n)
  fwd = resample_transform(T, m, σ, C)
  Resample{T,typeof(fwd),typeof(ws)}(n, corr, ws, fwd, Vector{Complex{T}}(undef, m),
                                     Vector{Complex{T}}(undef, m), Vector{Complex{T}}(undef, m))
end

resample_transform(::Type{T}, m::Int, σ::Int, ::Type{C}) where {T,C} = C{T}(m, σ)
resample_transform(::Type{T}, m::Int, σ::Int, ::Type{TypedKernels}) where {T} =
  m * sizeof(Complex{T}) >= BLUESTEIN_FOURSTEP_BYTES && fourstep_split(m) > 0 ? FourStep{T}(m, σ, 1) : RChain{T}(m, σ)

# Grid length, width, deconvolution factors and weights of a length-n resampling (shared by plans).
function resample_parts(::Type{T}, n::Int, m::Int = bluestein_length(T, 2n)) where {T}
  w = resample_width(T)
  corr = cached(() -> resample_corr(T, n, m, w), CacheKey(:resample_corr, T, n, 0, [m, w]))::Vector{T}
  ws = n * 2w * sizeof(T) <= resample_table_bytes(T) ?
       cached(() -> resample_table(T, n, m, w), CacheKey(:resample_table, T, n, 0, [m, w]))::RTable{T} :
       cached(() -> resample_poly(T, n, m, w), CacheKey(:resample_poly, T, n, 0, [m, w]))::RPoly{T}
  (m, w, corr, ws)
end

"""
    RsBatch{T}

The length-q DFTs of the s lanes of a composite kernel in one pass (FFTJ's batched resampling):
the grid holds rows of C lanes, lane fastest, so the transform is one chain batched over the
lanes and every output row is w FMAs of whole grid rows with broadcast weights (no reductions).
Lanes in chunks of C, so that a chunk's grid stays in cache.
"""
struct RsBatch{T,F,WS}
  n::Int
  m::Int
  C::Int
  corr::Vector{T}
  ws::WS
  fwd::F                          # length m over C lanes, sign σ
  g::Vector{Complex{T}}           # grid, m·C
  work1::Vector{Complex{T}}
  work2::Vector{Complex{T}}
end

# grid bytes of one lane chunk (as FFTJ's chunk budget)
const RS_CHUNK_BYTES = 1 << 18

function RsBatch{T}(q::Int, σ::Int, s::Int) where {T}
  m, _, corr, ws = resample_parts(T, q)
  CU = ncomplex(first(vec_ladder(T)))
  C = min(s, max(CU, CU * fld(RS_CHUNK_BYTES ÷ (m * sizeof(Complex{T})), CU)))
  fwd = RChain{T}(m, σ, radices(m, C; esize = sizeof(Complex{T})))
  RsBatch{T,typeof(fwd),typeof(ws)}(q, m, C, corr, ws, fwd, Vector{Complex{T}}(undef, m * C),
                                      Vector{Complex{T}}(undef, m * C), Vector{Complex{T}}(undef, m * C))
end

es_kernel(z, β) = abs(z) >= 1 ? zero(z) : exp(β * (sqrt(1 - z^2) - 1))
resample_beta(n, m, w) = 0.97π * w * (1 - n / 2m)

# Gauss-Legendre nodes and weights on [-1, 1] (Newton on the three-term recurrence)
function gauss_legendre(nq::Int)
  z = zeros(nq); wq = zeros(nq)
  for i in 1:nq
    x = cospi((i - 0.25) / (nq + 0.5))
    dp = 1.0
    for _ in 1:20
      p0, p1 = 1.0, x
      for k in 2:nq
        p0, p1 = p1, ((2k - 1) * x * p1 - (k - 1) * p0) / k
      end
      dp = nq * (x * p1 - p0) / (x^2 - 1)
      dx = p1 / dp
      x -= dx
      abs(dx) < 4eps() && break
    end
    z[i] = x
    wq[i] = 2 / ((1 - x^2) * dp^2)
  end
  z, wq
end

# Φ̂(ν) = w·∫₀¹ψ(z)cos(πνwz)dz by the positive half of a symmetric Gauss-Legendre rule
function resample_corr(::Type{T}, n::Int, m::Int, w::Int) where {T}
  β = resample_beta(n, m, w)
  zq, wq = gauss_legendre(2w + 20)
  hp = cld(n, 2)
  corr = Vector{T}(undef, n)
  for j in 0:n-1
    ν = (j < hp ? j : j - n) / m
    Φ = 0.0
    for q in eachindex(zq)
      zq[q] > 0 && (Φ += wq[q] * es_kernel(zq[q], β) * cospi(ν * w * zq[q]))
    end
    corr[j+1] = T(1 / (w * Φ))
  end
  corr
end

function resample_table(::Type{T}, n::Int, m::Int, w::Int) where {T}
  β = resample_beta(n, m, w)
  h = w ÷ 2
  wt = Vector{T}(undef, n * 2w)
  l0 = Vector{Int32}(undef, n)
  for k in 0:n-1
    q, r = divrem(widemul(k, m), n)
    l0[k+1] = Int32(q - h + 1)
    δ = r / n
    for i in 0:w-1
      wt[k*2w+2i+1] = wt[k*2w+2i+2] = T(es_kernel((δ + h - 1 - i) / h, β))
    end
  end
  RTable{T}(wt, l0)
end

# Chebyshev interpolants of each tap's weight on each piece, in monomials of the local variable
# t ∈ [-1, 1]; the degree grows until the fit error is ≤ tol/16 or stops halving (coefficient
# rounding floors it).
function resample_poly(::Type{T}, n::Int, m::Int, w::Int) where {T}
  β = resample_beta(n, m, w)
  P = sizeof(T) >= 8 ? 32 : 64
  tolw = resample_tol(T) / 16
  prev, preverr = resample_fit(T, β, w, P, 1)
  d = 1
  for dd in 2:30
    C, err = resample_fit(T, β, w, P, dd)
    if err <= tolw
      prev, d = C, dd
      break
    end
    err > preverr / 2 && break
    prev, preverr, d = C, err, dd
  end
  RPoly{T}(prev, P, d)
end

function resample_fit(::Type{T}, β, w::Int, P::Int, d::Int) where {T}
  h = w ÷ 2
  tj = [cospi((j + 0.5) / (d + 1)) for j in 0:d]
  C = zeros(T, P * (d + 1) * w)
  mono = zeros(d + 1)
  err = 0.0
  for p in 0:P-1, i in 0:w-1
    f(t) = es_kernel(((p + (t + 1) / 2) / P + h - 1 - i) / h, β)
    fj = f.(tj)
    fill!(mono, 0)
    Tkm, Tk = zeros(d + 1), zeros(d + 1)   # T₀, T₁ in monomials
    Tkm[1] = 1
    Tk[2] = 1
    for k in 0:d
      a = sum(fj[j+1] * cospi(k * (j + 0.5) / (d + 1)) for j in 0:d) * 2 / (d + 1)
      k == 0 && (a /= 2)
      mono .+= a .* (k == 0 ? Tkm : Tk)
      if k >= 1                      # T_{k+1} = 2t·T_k - T_{k-1}
        Tn = -Tkm
        Tn[2:end] .+= 2 .* Tk[1:end-1]
        Tkm, Tk = Tk, Tn
      end
    end
    for c in 0:d
      mono[c+1] = C[(p*(d+1)+c)*w+i+1] = T(mono[c+1])
    end
    for t in range(-1.0, 1.0; length = 4d + 4)
      err = max(err, abs(evalpoly(t, mono) - f(t)))
    end
  end
  C, err
end

# Grid start (0-based, unwrapped) and weights of output k. The table needs no state; the
# polynomials step st = (q, r), k·m = q·n + r, by (mq, mr) = divrem(m, n) per output, from output k.
rstart(::RTable, k, m, n) = nothing
rstart(::RPoly, k, m, n) = divrem(k * m, n)
@inline radvance(::RTable, st, mq, mr, n) = st
@inline function radvance(::RPoly, (q, r), mq, mr, n)
  r += mr
  c = r >= n
  (q + mq + c, r - ifelse(c, n, 0))
end
# all U windows from output k inside the grid (for the polynomials a bound on the last start: each
# output advances q by at most mq + 1)
@inline rblock_ok(ws::RTable, k, st, U, w, m, mq) =
  @inbounds ws.l0[k+1] >= 0 && ws.l0[k+U] + w <= m
@inline rblock_ok(::RPoly, k, (q, r), U, w, m, mq) =
  q - w ÷ 2 + 1 >= 0 && q + (U - 1) * (mq + 1) + w ÷ 2 + 1 <= m

@inline rweights(ws::RTable{T}, k, st, n, ::Val{w}) where {T,w} =
  @inbounds (Int(ws.l0[k+1]), vload(Vec{2w,T}, pointer(ws.wt) + k * 2w * sizeof(T)))
@inline function rweights(ws::RPoly{T}, k, (q, r), n, ::Val{w}) where {T,w}
  d = ws.d
  s = r * (ws.P / n)                 # δ·P ∈ [0, P)
  pc = min(unsafe_trunc(Int, s), ws.P - 1)
  t = T(2 * (s - pc) - 1)
  pC = pointer(ws.C) + pc * (d + 1) * w * sizeof(T)
  wv = vload(Vec{w,T}, pC + d * w * sizeof(T))
  for c in d-1:-1:0
    wv = muladd(wv, t, vload(Vec{w,T}, pC + c * w * sizeof(T)))
  end
  (q - w ÷ 2 + 1, shufflevector(wv, Val(ntuple(l -> (l - 1) ÷ 2, Val(2w)))))
end

# Lane by lane: deconvolve and zero-pad, transform, interpolate. Element j of lane i at
# px + i + Sx·j, py + i + Sy·j. Uses the kernel's own buffers (one thread at a time).
function run_lanes!(py::Ptr{T}, px::Ptr{T}, ::Ptr{T}, ::Ptr{T}, P::Resample{T}, Sx::Int, Sy::Int,
                    B::Int) where {T}
  n = P.n; buf = P.buf; m = length(buf); hp = cld(n, 2); corr = P.corr
  cx = Ptr{Complex{T}}(px); cy = Ptr{Complex{T}}(py)
  GC.@preserve P begin
    pb = cptr(buf)
    for i in 1:B
      @inbounds for j in 1:hp
        buf[j] = unsafe_load(cx, i + Sx * (j - 1)) * corr[j]
      end
      @inbounds for j in hp+1:n
        buf[m-n+j] = unsafe_load(cx, i + Sx * (j - 1)) * corr[j]
      end
      fill!(view(buf, hp+1:m-n+hp), zero(Complex{T}))
      run!(pb, pb, cptr(P.work1), cptr(P.work2), P.fwd, 1)
      resample_interp!(cy, i, Sy, pb, P.ws, n, m, Val(resample_width(T)), 0, n)
    end
  end
  nothing
end

run!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::Resample{T}, S::Int) where {T} =
  run_lanes!(py, px, pw1, pw2, P, S, S, S)

# The length-q DFT of a composite kernel (see the Bluestein method in plan.jl): twiddles and
# deconvolution in one pass, then interpolation straight into the strided output.
function twiddled_leftover!(py::Ptr{T}, pz::Ptr{T}, tw::Vector{Complex{T}}, rs::Resample{T}, n::Int, s::Int,
                            B::Int, i::Int, k1::Int, Sy::Int) where {T}
  q = rs.n; buf = rs.buf; M = length(buf); hp = cld(q, 2); corr = rs.corr
  cz = Ptr{Complex{T}}(pz); cy = Ptr{Complex{T}}(py)
  GC.@preserve rs tw begin
    pb = cptr(buf)
    zb = i + B * q * k1 + 1
    e = 0                           # (j2·k1) mod n
    @inbounds for j in 1:hp
      buf[j] = unsafe_load(cz, zb + B * (j - 1)) * tw[e+1] * corr[j]
      e += k1; e >= n && (e -= n)
    end
    @inbounds for j in hp+1:q
      buf[M-q+j] = unsafe_load(cz, zb + B * (j - 1)) * tw[e+1] * corr[j]
      e += k1; e >= n && (e -= n)
    end
    fill!(view(buf, hp+1:M-q+hp), zero(Complex{T}))
    run!(pb, pb, cptr(rs.work1), cptr(rs.work2), rs.fwd, 1)
    resample_interp!(cy, i + Sy * k1 + 1, Sy * s, pb, rs.ws, q, M, Val(resample_width(T)), 0, q)
  end
  nothing
end

# Tree reduction of U vectors of U complex values, each vector one output's partial sums, into one
# vector of the U outputs: per level, pairs of vectors halve their segments (FFTJ's interleaved
# reduction with complex values as units). Returns the statements and the result in output order.
function tree_reduce_units(vecs, U)
  labels = [fill(j, U) for j in 0:U-1]
  lanes(us) = Tuple(l for u in us for l in (2u, 2u + 1))
  stmts = Any[]
  L = U
  while L > 1
    half = L ÷ 2
    nvecs = Any[]; nlabels = Vector{Int}[]
    for p in 1:2:length(vecs)
      ip = Int[]; iq = Int[]; lab = Int[]
      for u in 0:U-1
        seg, pos = divrem(u, L)
        first = pos < half
        sw = seg * L + (pos + half) % L
        push!(ip, first ? u : U + u)
        push!(iq, first ? sw : U + sw)
        push!(lab, first ? labels[p][u+1] : labels[p+1][u+1])
      end
      t = gensym(:t)
      A, B = vecs[p], vecs[p+1]
      push!(stmts, :($t = shufflevector($A, $B, Val($(lanes(ip)))) + shufflevector($A, $B, Val($(lanes(iq))))))
      push!(nvecs, t); push!(nlabels, lab)
    end
    vecs, labels = nvecs, nlabels
    L = half
  end
  perm = [findfirst(==(u), labels[1]) - 1 for u in 0:U-1]
  res = perm == 0:U-1 ? vecs[1] : :(shufflevector($(vecs[1]), Val($(lanes(perm)))))
  stmts, res
end

# Output k = Σ_t weight_t · grid[(l + t) mod m]. Blocks of U = V/2 outputs (V lanes in the widest
# vector) whose windows lie inside the grid: per output one load of the w interleaved grid values,
# folded with the weights into V lanes by FMAs, then one tree reduction for the block and one
# store. Other outputs one at a time. Outputs k0 … k1-1 (a range of them per task on threads).
@generated function resample_interp!(cy::Ptr{Complex{T}}, i::Int, Sy::Int, pb::Ptr{T}, ws::WS, n::Int,
                                     m::Int, ::Val{w}, k0::Int, k1::Int) where {T,WS,w}
  V = 2 * ncomplex(first(vec_ladder(T)))
  U = V ÷ 2
  ch(v, c) = :(shufflevector($v, Val($(Tuple(c*V:c*V+V-1)))))
  fold(g, wv) = foldl((acc, c) -> :(muladd($(ch(g, c)), $(ch(wv, c)), $acc)), 1:2w÷V-1;
                      init = :($(ch(g, 0)) * $(ch(wv, 0))))
  blk = Any[]
  fs = Any[]
  for j in 0:U-1
    f = gensym(:f)
    push!(blk, quote
      l, wv = rweights(ws, k + $j, st, n, Val($w))
      g = vload(Vec{$(2w),T}, pb + l * $(2sizeof(T)))
      $f = $(fold(:g, :wv))
      st = radvance(ws, st, mq, mr, n)
    end)
    push!(fs, f)
  end
  stmts, res = tree_reduce_units(fs, U)
  blocked = U >= 2 && (2w) % V == 0
  quote
    mq, mr = divrem(m, n)
    st = rstart(ws, k0, m, n)
    k = k0
    GC.@preserve ws @inbounds while k < k1
      if $blocked && k + $U <= k1 && rblock_ok(ws, k, st, $U, $w, m, mq)
        $(blk...)
        $(stmts...)
        if Sy == 1
          vstore($res, Ptr{T}(cy) + (i - 1 + k) * $(2sizeof(T)))
        else
          v = $res
          for u in 0:$(U - 1)
            unsafe_store!(cy, Complex{T}(v[2u+1], v[2u+2]), i + Sy * (k + u))
          end
        end
        k += $U
        continue
      end
      l, wv = rweights(ws, k, st, n, Val($w))
      if l >= 0 && l + $w <= m
        p = vload(Vec{$(2w),T}, pb + l * $(2sizeof(T))) * wv
        acc = Complex{T}(sum(shufflevector(p, Val($(Tuple(0:2:2w-1))))), sum(shufflevector(p, Val($(Tuple(1:2:2w-1))))))
      else                          # window wraps around the grid ends
        acc = zero(Complex{T})
        for t in 0:$(w - 1)
          g = l + t                 # l > -w and l + t < m + w: one correction, no division
          g += ifelse(g < 0, m, 0)
          g -= ifelse(g >= m, m, 0)
          acc += wv[2t+1] * unsafe_load(Ptr{Complex{T}}(pb), g + 1)
        end
      end
      unsafe_store!(cy, acc, i + Sy * k)
      st = radvance(ws, st, mq, mr, n)
      k += 1
    end
    nothing
  end
end

# The composite kernel's length-q DFTs by the batched pass: for each slab lane i, the s lanes k1
# (z element j2 at i + B·(j2 + q·k1)) chunk by chunk into the grid with the twiddles w_n^(j2·k1)
# and the deconvolution, one batched transform, then rows of output (y + i + Sy·(k1 + s·k2)).
function leftover!(py::Ptr{T}, pz::Ptr{T}, tw::Vector{Complex{T}}, R::RsBatch{T}, n::Int, s::Int,
                   B::Int, Sy::Int) where {T}
  for i in 0:B-1, c0 in 0:R.C:s-1
    rs_chunk!(py, pz, tw, R, n, s, B, Sy, i, c0, min(R.C, s - c0))
  end
  nothing
end

# Lanes c0 … c0+L-1 (L ≤ C, c0 a multiple of the vector's complex count CU) of slab lane i: one
# chunk of the batched pass, its multiple of CU lanes and its remainder as separate groups. The
# chain's vector width follows the lane count, and vectors of one complex value round differently;
# this way only the last s % CU lanes ever take the narrow path, so the threaded composite kernel
# (other chunk sizes, multiples of CU) agrees bitwise with the serial one.
function rs_chunk!(py::Ptr{T}, pz::Ptr{T}, tw::Vector{Complex{T}}, R::RsBatch{T}, n::Int, s::Int, B::Int,
                   Sy::Int, i::Int, c0::Int, L::Int) where {T}
  r = L % ncomplex(first(vec_ladder(T)))
  L > r && rs_group!(py, pz, tw, R, n, s, B, Sy, i, c0, L - r)
  r > 0 && rs_group!(py, pz, tw, R, n, s, B, Sy, i, c0 + L - r, r)
  nothing
end

function rs_group!(py::Ptr{T}, pz::Ptr{T}, tw::Vector{Complex{T}}, R::RsBatch{T}, n::Int, s::Int, B::Int,
                   Sy::Int, i::Int, c0::Int, L::Int) where {T}
  q = R.n; m = R.m; hp = cld(q, 2); corr = R.corr; g = R.g
  cz = Ptr{Complex{T}}(pz); cy = Ptr{Complex{T}}(py)
  GC.@preserve R tw begin
    pg = cptr(g)
    @inbounds for j2 in 0:q-1
      jj = j2 < hp ? j2 : m - q + j2
      e = (j2 * c0) % n             # (j2·k1) mod n, stepped over the lanes
      cr = corr[j2+1]
      zb = i + B * (j2 + q * c0) + 1
      for c in 0:L-1
        g[jj*L+c+1] = unsafe_load(cz, zb + B * q * c) * (tw[e+1] * cr)
        e += j2; e >= n && (e -= n)
      end
    end
    fill!(view(g, hp*L+1:(m-q+hp)*L), zero(Complex{T}))
    run!(pg, pg, cptr(R.work1), cptr(R.work2), R.fwd, L)
    rs_rows!(cy, i + Sy * c0 + 1, Sy, Sy * s, pg, R.ws, q, m, L, Val(resample_width(T)))
  end
  nothing
end

# Output rows of the batched pass: y[base + Ss·k2 + Sy·c] = Σ_t weight_t · grid row (l + t) mod m,
# lane c; whole vectors of lanes when Sy = 1, the rest one lane at a time.
@generated function rs_rows!(cy::Ptr{Complex{T}}, base::Int, Sy::Int, Ss::Int, pg::Ptr{T}, ws::WS, q::Int,
                             m::Int, L::Int, ::Val{w}) where {T,WS,w}
  CU = ncomplex(first(vec_ladder(T)))
  VT = Vec{2CU,T}
  CT = 2sizeof(T)
  o = [Symbol(:o, t) for t in 0:w-1]
  b = [Symbol(:b, t) for t in 0:w-1]
  offs = [:($(o[t+1]) = (g = l + $t; g += ifelse(g < 0, m, 0); g -= ifelse(g >= m, m, 0); g * L)) for t in 0:w-1]
  bcast = [:($(b[t+1]) = $VT(wv[$(2t + 1)])) for t in 0:w-1]
  vsum = foldl((acc, t) -> :(muladd($(b[t+1]), vload($VT, pg + ($(o[t+1]) + c) * $CT), $acc)), 1:w-1;
               init = :($(b[1]) * vload($VT, pg + ($(o[1]) + c) * $CT)))
  ssum = foldl((acc, t) -> :($acc + wv[$(2t + 1)] * unsafe_load(Ptr{Complex{T}}(pg), $(o[t+1]) + c + 1)), 1:w-1;
               init = :(wv[1] * unsafe_load(Ptr{Complex{T}}(pg), $(o[1]) + c + 1)))
  quote
    mq, mr = divrem(m, q)
    st = rstart(ws, 0, m, q)
    GC.@preserve ws @inbounds for k2 in 0:q-1
      l, wv = rweights(ws, k2, st, q, Val($w))
      $(offs...)
      yo = base + Ss * k2
      c = 0
      if Sy == 1
        $(bcast...)
        while c + $CU <= L
          vstore($vsum, Ptr{T}(cy) + (yo - 1 + c) * $CT)
          c += $CU
        end
      end
      while c < L
        unsafe_store!(cy, $ssum, yo + Sy * c)
        c += 1
      end
      st = radvance(ws, st, mq, mr, q)
    end
    nothing
  end
end
