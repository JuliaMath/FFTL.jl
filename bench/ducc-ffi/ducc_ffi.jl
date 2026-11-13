# Julia binding of libducc_ffi (ducc_ffi.cc, built with build.sh against the ducc0 sources). This
# replaces Ducc0.jl in every benchmark. Each *_setup returns (closure, planning time) like the
# library setups of overview/run_kinds.jl; ducc0 has no plan object for N-D transforms, so their
# time is the free call.
#
# Shapes are passed in C order: Julia's size, reversed, so the last C axis is Julia's first axis.

const LIB = joinpath(@__DIR__, Sys.isapple() ? "libducc_ffi.dylib" : "libducc_ffi.so")
# Without the built library the ducc0 competitor is left out (measure.jl says so).
const DUCC_AVAILABLE = isfile(LIB)

cshape(a) = Csize_t[reverse(size(a))...]

function ducc_c2c_setup(x::Array{ComplexF64}, y::Array{ComplexF64}, nt)
  sh = cshape(x); nd = Csize_t(length(sh)); nt = Csize_t(nt)
  (() -> ccall((:dfft_c2c_f64, LIB), Cvoid, (Ptr{ComplexF64}, Ptr{ComplexF64}, Ptr{Csize_t}, Csize_t, Cdouble, Cint, Csize_t),
               x, y, sh, nd, 1.0, Cint(1), nt)), 0.0
end
function ducc_c2c_setup(x::Array{ComplexF32}, y::Array{ComplexF32}, nt)
  sh = cshape(x); nd = Csize_t(length(sh)); nt = Csize_t(nt)
  (() -> ccall((:dfft_c2c_f32, LIB), Cvoid, (Ptr{ComplexF32}, Ptr{ComplexF32}, Ptr{Csize_t}, Csize_t, Cfloat, Cint, Csize_t),
               x, y, sh, nd, 1.0f0, Cint(1), nt)), 0.0
end

function ducc_r2c_setup(x::Array{Float64}, y::Array{ComplexF64}, nt)
  sh = cshape(x); nd = Csize_t(length(sh)); nt = Csize_t(nt)
  (() -> ccall((:dfft_r2c_f64, LIB), Cvoid, (Ptr{Float64}, Ptr{ComplexF64}, Ptr{Csize_t}, Csize_t, Cdouble, Csize_t),
               x, y, sh, nd, 1.0, nt)), 0.0
end
function ducc_r2c_setup(x::Array{Float32}, y::Array{ComplexF32}, nt)
  sh = cshape(x); nd = Csize_t(length(sh)); nt = Csize_t(nt)
  (() -> ccall((:dfft_r2c_f32, LIB), Cvoid, (Ptr{Float32}, Ptr{ComplexF32}, Ptr{Csize_t}, Csize_t, Cfloat, Csize_t),
               x, y, sh, nd, 1.0f0, nt)), 0.0
end

# c2r: `y` is the real output; its shape is the transform shape.
function ducc_c2r_setup(x::Array{ComplexF64}, y::Array{Float64}, nt)
  sh = cshape(y); nd = Csize_t(length(sh)); nt = Csize_t(nt)
  (() -> ccall((:dfft_c2r_f64, LIB), Cvoid, (Ptr{ComplexF64}, Ptr{Float64}, Ptr{Csize_t}, Csize_t, Cdouble, Csize_t),
               x, y, sh, nd, 1.0, nt)), 0.0
end
function ducc_c2r_setup(x::Array{ComplexF32}, y::Array{Float32}, nt)
  sh = cshape(y); nd = Csize_t(length(sh)); nt = Csize_t(nt)
  (() -> ccall((:dfft_c2r_f32, LIB), Cvoid, (Ptr{ComplexF32}, Ptr{Float32}, Ptr{Csize_t}, Csize_t, Cfloat, Csize_t),
               x, y, sh, nd, 1.0f0, nt)), 0.0
end

# --- 1-D plans (plan made once, buffer made once; each call copies its input first) ----------
# Returns (closure, planning seconds) or nothing. The copy of the input is inside the closure: the
# plan works in place. Real transforms convert between halfcomplex and complex in the library.
function ducc_plan1d_setup(kind::Symbol, x, y, nt)
  kind in (:c2c, :r2c, :c2r) || return nothing
  n = kind === :c2r ? length(y) : length(x)
  ducc_plan1d_setup(Val(kind), x, y, n, nt)
end

function ducc_plan1d_setup(::Val{:c2c}, x::Array{ComplexF64}, y::Array{ComplexF64}, n, nt)
  t = @elapsed p = ccall((:d1_plan_f64, LIB), Ptr{Cvoid}, (Csize_t,), n)
  buf = Vector{ComplexF64}(undef, Int(ccall((:d1_bufsize_f64, LIB), Csize_t, (Ptr{Cvoid},), p)))
  xw = similar(x); h = Ref(p)
  finalizer(r -> ccall((:d1_free_f64, LIB), Cvoid, (Ptr{Cvoid},), r[]), h)
  (() -> (copyto!(xw, x); ccall((:d1_exec_f64, LIB), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cdouble, Cint, Csize_t),
                                 h[], xw, buf, 1.0, Cint(1), Csize_t(nt)))), t
end
function ducc_plan1d_setup(::Val{:c2c}, x::Array{ComplexF32}, y::Array{ComplexF32}, n, nt)
  t = @elapsed p = ccall((:d1_plan_f32, LIB), Ptr{Cvoid}, (Csize_t,), n)
  buf = Vector{ComplexF32}(undef, Int(ccall((:d1_bufsize_f32, LIB), Csize_t, (Ptr{Cvoid},), p)))
  xw = similar(x); h = Ref(p)
  finalizer(r -> ccall((:d1_free_f32, LIB), Cvoid, (Ptr{Cvoid},), r[]), h)
  (() -> (copyto!(xw, x); ccall((:d1_exec_f32, LIB), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cfloat, Cint, Csize_t),
                                 h[], xw, buf, 1.0f0, Cint(1), Csize_t(nt)))), t
end

function ducc_plan1d_setup(::Val{:r2c}, x::Array{Float64}, y::Array{ComplexF64}, n, nt)
  t = @elapsed p = ccall((:d1r_plan_f64, LIB), Ptr{Cvoid}, (Csize_t,), n)
  buf = Vector{Float64}(undef, Int(ccall((:d1r_bufsize_f64, LIB), Csize_t, (Ptr{Cvoid},), p)))
  xw = similar(x); h = Ref(p)
  finalizer(r -> ccall((:d1r_free_f64, LIB), Cvoid, (Ptr{Cvoid},), r[]), h)
  (() -> (copyto!(xw, x); ccall((:d1r_r2c_f64, LIB), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Csize_t),
                                 h[], xw, buf, y, n, Csize_t(nt)))), t
end
function ducc_plan1d_setup(::Val{:r2c}, x::Array{Float32}, y::Array{ComplexF32}, n, nt)
  t = @elapsed p = ccall((:d1r_plan_f32, LIB), Ptr{Cvoid}, (Csize_t,), n)
  buf = Vector{Float32}(undef, Int(ccall((:d1r_bufsize_f32, LIB), Csize_t, (Ptr{Cvoid},), p)))
  xw = similar(x); h = Ref(p)
  finalizer(r -> ccall((:d1r_free_f32, LIB), Cvoid, (Ptr{Cvoid},), r[]), h)
  (() -> (copyto!(xw, x); ccall((:d1r_r2c_f32, LIB), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Csize_t),
                                 h[], xw, buf, y, n, Csize_t(nt)))), t
end

function ducc_plan1d_setup(::Val{:c2r}, x::Array{ComplexF64}, y::Array{Float64}, n, nt)
  t = @elapsed p = ccall((:d1r_plan_f64, LIB), Ptr{Cvoid}, (Csize_t,), n)
  buf = Vector{Float64}(undef, Int(ccall((:d1r_bufsize_f64, LIB), Csize_t, (Ptr{Cvoid},), p)))
  h = Ref(p)
  finalizer(r -> ccall((:d1r_free_f64, LIB), Cvoid, (Ptr{Cvoid},), r[]), h)
  (() -> ccall((:d1r_c2r_f64, LIB), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Csize_t),
               h[], x, y, buf, n, Csize_t(nt))), t
end
function ducc_plan1d_setup(::Val{:c2r}, x::Array{ComplexF32}, y::Array{Float32}, n, nt)
  t = @elapsed p = ccall((:d1r_plan_f32, LIB), Ptr{Cvoid}, (Csize_t,), n)
  buf = Vector{Float32}(undef, Int(ccall((:d1r_bufsize_f32, LIB), Csize_t, (Ptr{Cvoid},), p)))
  h = Ref(p)
  finalizer(r -> ccall((:d1r_free_f32, LIB), Cvoid, (Ptr{Cvoid},), r[]), h)
  (() -> ccall((:d1r_c2r_f32, LIB), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Csize_t),
               h[], x, y, buf, n, Csize_t(nt))), t
end

# --- real-to-real by FFTW kind code (R2HC 0, HC2R 1, DHT 2, REDFT00 3 … RODFT11 10) ------------
function ducc_r2r_setup(x::Array{Float64}, y::Array{Float64}, nt, code::Integer)
  sh = cshape(x); nd = Csize_t(length(sh)); c = Cint(code); nt = Csize_t(nt)
  (() -> ccall((:dfft_r2r_f64, LIB), Cvoid, (Cint, Ptr{Csize_t}, Csize_t, Ptr{Float64}, Ptr{Float64}, Csize_t),
               c, sh, nd, x, y, nt)), 0.0
end
function ducc_r2r_setup(x::Array{Float32}, y::Array{Float32}, nt, code::Integer)
  sh = cshape(x); nd = Csize_t(length(sh)); c = Cint(code); nt = Csize_t(nt)
  (() -> ccall((:dfft_r2r_f32, LIB), Cvoid, (Cint, Ptr{Csize_t}, Csize_t, Ptr{Float32}, Ptr{Float32}, Csize_t),
               c, sh, nd, x, y, nt)), 0.0
end
