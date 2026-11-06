# FFTL as a backend of AbstractFFTs.jl's generic functions.
#
# `FFTL.fft`, `FFTL.plan_fft`, … are FFTL's own functions (src/interface.jl, src/real.jl,
# src/r2r.jl), so code that writes `using FFTL` reaches FFTL whatever else is loaded. Code that
# calls AbstractFFTs's generic functions (`AbstractFFTs.fft(x)`, or `fft` after
# `using AbstractFFTs`) reaches FFTL through the methods below. Their signatures are
# deliberately less specific than FFTW.jl's (`StridedArray` of Float32/Float64 complex or real
# values): when FFTW.jl is loaded too, the generic functions take FFTW for those arrays, and
# FFTL for the rest of the floating-point types and array types. Methods with FFTW.jl's exact
# signatures would overwrite each other depending on the load order and break the
# precompilation of every package that depends on both.
#
# AbstractFFTs derives the rest from these: the real-to-complex promotion of `fft`, the
# normalised inverses (`plan_ifft`, `plan_irfft`, `plan_ifft!`) and the transform functions.

AbstractFFTs.plan_fft(x::AbstractArray{<:Complex{<:AbstractFloat}}, region; kw...) = plan_fft(x, region; kw...)
AbstractFFTs.plan_bfft(x::AbstractArray{<:Complex{<:AbstractFloat}}, region; kw...) = plan_bfft(x, region; kw...)
AbstractFFTs.plan_fft!(x::AbstractArray{<:Complex{<:AbstractFloat}}, region; kw...) = plan_fft!(x, region; kw...)
AbstractFFTs.plan_bfft!(x::AbstractArray{<:Complex{<:AbstractFloat}}, region; kw...) = plan_bfft!(x, region; kw...)
AbstractFFTs.plan_rfft(x::AbstractArray{<:AbstractFloat}, region; kw...) = plan_rfft(x, region; kw...)
AbstractFFTs.plan_brfft(x::AbstractArray{<:Complex{<:AbstractFloat}}, d::Integer, region; kw...) =
  plan_brfft(x, d, region; kw...)
