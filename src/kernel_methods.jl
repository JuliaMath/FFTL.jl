# The methods of the stage kernels: one per kernel of STAGE_KERNELS (kernels.jl), element type,
# sign, vector type of the ladder and radix of RADICES, the set that `RChain` dispatches to. They
# are plain methods: a package image keeps the complete optimised IR of each precompiled
# specialization of a `@generated` method but none of a plain one.
for T in (Float32, Float64), (name, args, body) in STAGE_KERNELS
  ladder = vec_ladder(T)
  typed = [a === :pt ? :($a::Ptr{$T}) : :($a::Int) for a in args]
  for σ in (-1, 1), (i, V) in enumerate(ladder), R in RADICES
    kernel_exists(name, ladder, i, R) || continue
    @eval function $name(py::Ptr{$T}, px::Ptr{$T}, $(typed...), ::Val{$R}, ::Val{$σ}, ::Type{$V})
      VT = $V             # the codelet statements name the vector type
      $(body(T, R, σ, V))
    end
  end
end
