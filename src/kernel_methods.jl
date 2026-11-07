# Compiled-ahead methods of the stage kernels for the fixed codelet set (radix × sign × vector
# type) that `RChain` dispatches to, and of the chains of the small lengths (rchain.jl). The
# `@generated` kernels in kernels.jl define every other combination. A package image keeps the
# complete optimised IR of each precompiled `@generated` specialization but none of a plain
# method, so the fixed set is defined as plain methods with the same bodies. The codelets of
# LAZY_RADICES are not in the set.

kernel_subst(x, d) = x isa Symbol ? get(d, x, x) :
                     x isa Expr ? Expr(x.head, map(a -> kernel_subst(a, d), x.args)...) : x

const KERNEL_HEADS = (
  (:stage!, :(stage!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, s::Int, plo::Int, phi::Int,
                     ::Val{R}, ::Val{σ}, ::Type{VT})), :all),
  (:stage_range!, :(stage_range!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, s::Int, qlo::Int, qhi::Int,
                                 plo::Int, phi::Int, ::Val{R}, ::Val{σ}, ::Type{VT})), :all),
  (:stage_last!, :(stage_last!(py::Ptr{T}, px::Ptr{T}, s::Int, qlo::Int, qhi::Int,
                               ::Val{R}, ::Val{σ}, ::Type{VT})), :all),
  (:stage_lanes!, :(stage_lanes!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, s0::Int, Sx::Int, Sy::Int,
                                 B::Int, plo::Int, phi::Int, ::Val{R}, ::Val{σ}, ::Type{VT})), :all),
  (:stage_last_lanes!, :(stage_last_lanes!(py::Ptr{T}, px::Ptr{T}, s0::Int, Sx::Int, Sy::Int, B::Int,
                                           qlo::Int, qhi::Int, ::Val{R}, ::Val{σ}, ::Type{VT})), :all),
  # the grouped first stage exists for 2 or 4 complex values per vector only; its scalar tail
  # runs on one complex value
  (:stage_first_group_range!, :(stage_first_group_range!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int,
                                plo::Int, phi::Int, ::Val{R}, ::Val{σ}, ::Type{VT})), :group),
  (:stage_first_group_1!, :(stage_first_group_1!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int,
                            plo::Int, phi::Int, ::Val{R}, ::Val{σ}, ::Type{VT})), :tail))

for T in (Float32, Float64), (name, head, vts) in KERNEL_HEADS
  body = getfield(@__MODULE__, Symbol(string(name)[1:end-1], "_body"))
  vts === :all ? (types = vec_ladder(T)) :
  vts === :group ? (types = filter(V -> ncomplex(V) in (2, 4), vec_ladder(T))) :
  (types = any(V -> ncomplex(V) in (2, 4), vec_ladder(T)) ? (Vec{2,T},) : ())
  for σ in (-1, 1), VT in types, R in RADICES
    R in LAZY_RADICES && continue
    h = kernel_subst(head, Dict{Symbol,Any}(:T => T, :R => R, :σ => σ, :VT => VT))
    # the codelet statements name the type parameters T and VT, which a plain method does not bind
    b = kernel_subst(body(T, R, σ, VT), Dict{Symbol,Any}(:T => T, :VT => VT))
    @eval $(Expr(:function, h, b))
  end
end

# stage_small!: the widest vector type, power-of-two radices, every stride s < C that divides C.
for T in (Float32, Float64), σ in (-1, 1), R in RADICES
  VT = first(vec_ladder(T)); C = ncomplex(VT)
  (ispow2(R) && !(R in LAZY_RADICES)) || continue
  for s in 1:C-1
    C % s == 0 || continue
    h = :(stage_small!(py::Ptr{$T}, px::Ptr{$T}, pt::Ptr{$T}, ms::Int, ::Val{$R}, ::Val{$σ}, ::Type{$VT}, ::Val{$s}))
    b = kernel_subst(stage_small_body(T, R, σ, VT, s), Dict{Symbol,Any}(:T => T, :VT => VT))
    @eval $(Expr(:function, h, b))
  end
end

# The compiled chains of the small lengths (rchain.jl) run through `run!`/`run_lanes!` methods of
# plain definition for the same reason; any other chain type uses the generic methods.
for T in (Float32, Float64), (n, σ) in SMALL_ENTRIES
  Rs = Tuple(radices(n, 1; esize = sizeof(Complex{T})))
  C = Chain{T,Rs,σ,length(Rs)}
  @eval $(Expr(:function, :(run!(py::Ptr{$T}, px::Ptr{$T}, pw1::Ptr{$T}, pw2::Ptr{$T}, P::$C, S::Int)),
               run_body(T, Rs, σ, length(Rs))))
  @eval $(Expr(:function, :(run_lanes!(py::Ptr{$T}, px::Ptr{$T}, pw1::Ptr{$T}, pw2::Ptr{$T}, P::$C,
                                       Sx::Int, Sy::Int, B::Int)),
               run_lanes_body(T, Rs, σ, length(Rs))))
end
