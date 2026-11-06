# The legacy Fortran interface of FFTW3 (the manual's chapter "Calling FFTW from Legacy
# Fortran") over the C interface of fftw3api.jl: dfftw_* in double and sfftw_* in single
# precision, each under the two name manglings the manual describes (one trailing underscore,
# the common convention, and two, g77's convention for names that contain an underscore).
#
# Fortran passes every argument by reference: scalars arrive as pointers. A function's return
# value becomes an extra first argument (the plan of a planner, the result of
# dfftw_init_threads, ...). A plan is stored in an integer*8 (the manual recommends it; any
# integer at least as wide as a pointer works). The basic and advanced planners take
# column-major (Fortran) dimensions and reverse them, together with inembed, onembed and the
# r2r kinds; the guru planners take the iodims as three integer arrays (n, is, os) and keep
# their order, except that dfftw_plan_guru_r2r reverses its kind array, as the manual states.
# The routines with FILE or string arguments, and memory allocation, have no legacy Fortran
# form (the manual's list), so the Fortran 2003 interface (include/fftw3.f03) binds to the C
# functions directly.

ld_int(p::Ptr{Cint}) = Int(unsafe_load(p))
ld_flags(p::Ptr{Cint}) = UInt32(reinterpret(UInt32, unsafe_load(p)))

# A plan in a Fortran integer*8, and back
ld_plan(p::Ptr{Int64}) = Ptr{Cvoid}(UInt(reinterpret(UInt64, unsafe_load(p))))
st_plan(p::Ptr{Int64}, q::Ptr{Cvoid}) = (unsafe_store!(p, reinterpret(Int64, UInt64(UInt(q)))); nothing)

function rev_ints(p::Ptr{Cint}, n::Int)
  n >= 0 || throw(ArgumentError("FFTW interface: rank must be nonnegative"))
  v = read_ints(p, n)
  reverse!(v)
end

function rev_kinds(p::Ptr{Cint}, n::Int)
  v = read_kinds_fftw(p, n)
  reverse!(v)
end

# The advanced planners: n, inembed and onembed in Fortran order. A null embed address (a
# compiler passing an absent argument) means NULL.
function fplan_many(::Type{T}, typ::Int, rank::Int, n::Ptr{Cint}, howmany::Int, pin::Ptr{Cvoid},
                    inembed::Ptr{Cint}, istride::Int, idist::Int, pout::Ptr{Cvoid}, onembed::Ptr{Cint},
                    ostride::Int, odist::Int, kinds::Ptr{Cint}, sign::Int, flags::UInt32) where {T}
  nv = rev_ints(n, rank)
  ie = inembed == C_NULL ? Int[] : rev_ints(inembed, rank)
  oe = onembed == C_NULL ? Int[] : rev_ints(onembed, rank)
  ks = typ == W_R2R ? rev_kinds(kinds, rank) : Int[]
  td, vd = many_dims(nv, howmany, ie, istride, idist, oe, ostride, odist, typ, pin == pout)
  wplan(T, typ, td, vd, ks, pin, pout, sign, flags)
end

function fiodims(rank::Int, n::Ptr{Cint}, is::Ptr{Cint}, os::Ptr{Cint})
  rank >= 0 || throw(ArgumentError("FFTW interface: ranks must be nonnegative"))
  NTuple{3,Int}[(Int(unsafe_load(n, i)), Int(unsafe_load(is, i)), Int(unsafe_load(os, i))) for i in 1:rank]
end

# The guru planners: iodims as three arrays; r2r kinds reversed.
function fplan_guru(::Type{T}, typ::Int, rank::Int, n::Ptr{Cint}, is::Ptr{Cint}, os::Ptr{Cint},
                    hrank::Int, hn::Ptr{Cint}, his::Ptr{Cint}, hos::Ptr{Cint}, pin::Ptr{Cvoid},
                    pout::Ptr{Cvoid}, kinds::Ptr{Cint}, sign::Int, flags::UInt32;
                    split::Bool = false, pin2::Ptr{Cvoid} = C_NULL, pout2::Ptr{Cvoid} = C_NULL) where {T}
  td = fiodims(rank, n, is, os)
  vd = fiodims(hrank, hn, his, hos)
  ks = typ == W_R2R ? rev_kinds(kinds, rank) : Int[]
  wplan(T, typ, td, vd, ks, pin, pout, sign, flags; split = split, pin2 = pin2, pout2 = pout2)
end

# Wisdom through a Fortran subroutine write_char(c, data): both by reference, data passed
# through as the address the program gave.
function fexport_callback(write_char::Ptr{Cvoid}, data::Ptr{Cvoid})
  c = Ref{Cchar}(0)
  for u in codeunits(W_WISDOM)
    c[] = u % Cchar
    ccall(write_char, Cvoid, (Ref{Cchar}, Ptr{Cvoid}), c, data)
  end
  nothing
end

const W_EXEC_FAILED = "FFTW interface: execution failed"

for (T, pre) in ((Float64, "dfftw_"), (Float32, "sfftw_")), suffix in ("_", "__")
  fname(name) = Symbol(pre, name, suffix)
  @eval begin
    # --- basic interface: dimensions in Fortran order
    Base.@ccallable function $(fname(:plan_dft_1d))(plan::Ptr{Int64}, n0::Ptr{Cint}, in_::Ptr{Cvoid}, out::Ptr{Cvoid},
                                                    sign::Ptr{Cint}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try wplan_basic($T, W_C2C, Int[ld_int(n0)], in_, out, Int[], ld_int(sign), ld_flags(flags)) catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_dft_2d))(plan::Ptr{Int64}, n0::Ptr{Cint}, n1::Ptr{Cint}, in_::Ptr{Cvoid},
                                                    out::Ptr{Cvoid}, sign::Ptr{Cint}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try wplan_basic($T, W_C2C, Int[ld_int(n1), ld_int(n0)], in_, out, Int[], ld_int(sign), ld_flags(flags)) catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_dft_3d))(plan::Ptr{Int64}, n0::Ptr{Cint}, n1::Ptr{Cint}, n2::Ptr{Cint},
                                                    in_::Ptr{Cvoid}, out::Ptr{Cvoid}, sign::Ptr{Cint}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try wplan_basic($T, W_C2C, Int[ld_int(n2), ld_int(n1), ld_int(n0)], in_, out, Int[], ld_int(sign), ld_flags(flags)) catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_dft))(plan::Ptr{Int64}, rank::Ptr{Cint}, n::Ptr{Cint}, in_::Ptr{Cvoid},
                                                 out::Ptr{Cvoid}, sign::Ptr{Cint}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try wplan_basic($T, W_C2C, rev_ints(n, ld_int(rank)), in_, out, Int[], ld_int(sign), ld_flags(flags)) catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_dft_r2c_1d))(plan::Ptr{Int64}, n0::Ptr{Cint}, in_::Ptr{Cvoid}, out::Ptr{Cvoid},
                                                        flags::Ptr{Cint})::Cvoid
      st_plan(plan, try wplan_basic($T, W_R2C, Int[ld_int(n0)], in_, out, Int[], -1, ld_flags(flags)) catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_dft_r2c_2d))(plan::Ptr{Int64}, n0::Ptr{Cint}, n1::Ptr{Cint}, in_::Ptr{Cvoid},
                                                        out::Ptr{Cvoid}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try wplan_basic($T, W_R2C, Int[ld_int(n1), ld_int(n0)], in_, out, Int[], -1, ld_flags(flags)) catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_dft_r2c_3d))(plan::Ptr{Int64}, n0::Ptr{Cint}, n1::Ptr{Cint}, n2::Ptr{Cint},
                                                        in_::Ptr{Cvoid}, out::Ptr{Cvoid}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try wplan_basic($T, W_R2C, Int[ld_int(n2), ld_int(n1), ld_int(n0)], in_, out, Int[], -1, ld_flags(flags)) catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_dft_r2c))(plan::Ptr{Int64}, rank::Ptr{Cint}, n::Ptr{Cint}, in_::Ptr{Cvoid},
                                                     out::Ptr{Cvoid}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try wplan_basic($T, W_R2C, rev_ints(n, ld_int(rank)), in_, out, Int[], -1, ld_flags(flags)) catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_dft_c2r_1d))(plan::Ptr{Int64}, n0::Ptr{Cint}, in_::Ptr{Cvoid}, out::Ptr{Cvoid},
                                                        flags::Ptr{Cint})::Cvoid
      st_plan(plan, try wplan_basic($T, W_C2R, Int[ld_int(n0)], in_, out, Int[], 1, ld_flags(flags)) catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_dft_c2r_2d))(plan::Ptr{Int64}, n0::Ptr{Cint}, n1::Ptr{Cint}, in_::Ptr{Cvoid},
                                                        out::Ptr{Cvoid}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try wplan_basic($T, W_C2R, Int[ld_int(n1), ld_int(n0)], in_, out, Int[], 1, ld_flags(flags)) catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_dft_c2r_3d))(plan::Ptr{Int64}, n0::Ptr{Cint}, n1::Ptr{Cint}, n2::Ptr{Cint},
                                                        in_::Ptr{Cvoid}, out::Ptr{Cvoid}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try wplan_basic($T, W_C2R, Int[ld_int(n2), ld_int(n1), ld_int(n0)], in_, out, Int[], 1, ld_flags(flags)) catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_dft_c2r))(plan::Ptr{Int64}, rank::Ptr{Cint}, n::Ptr{Cint}, in_::Ptr{Cvoid},
                                                     out::Ptr{Cvoid}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try wplan_basic($T, W_C2R, rev_ints(n, ld_int(rank)), in_, out, Int[], 1, ld_flags(flags)) catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_r2r_1d))(plan::Ptr{Int64}, n0::Ptr{Cint}, in_::Ptr{Cvoid}, out::Ptr{Cvoid},
                                                    k0::Ptr{Cint}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try wplan_basic($T, W_R2R, Int[ld_int(n0)], in_, out, Int[ld_int(k0)], 0, ld_flags(flags)) catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_r2r_2d))(plan::Ptr{Int64}, n0::Ptr{Cint}, n1::Ptr{Cint}, in_::Ptr{Cvoid},
                                                    out::Ptr{Cvoid}, k0::Ptr{Cint}, k1::Ptr{Cint}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try wplan_basic($T, W_R2R, Int[ld_int(n1), ld_int(n0)], in_, out, Int[ld_int(k1), ld_int(k0)], 0, ld_flags(flags)) catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_r2r_3d))(plan::Ptr{Int64}, n0::Ptr{Cint}, n1::Ptr{Cint}, n2::Ptr{Cint},
                                                    in_::Ptr{Cvoid}, out::Ptr{Cvoid}, k0::Ptr{Cint}, k1::Ptr{Cint},
                                                    k2::Ptr{Cint}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try wplan_basic($T, W_R2R, Int[ld_int(n2), ld_int(n1), ld_int(n0)], in_, out,
                                    Int[ld_int(k2), ld_int(k1), ld_int(k0)], 0, ld_flags(flags)) catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_r2r))(plan::Ptr{Int64}, rank::Ptr{Cint}, n::Ptr{Cint}, in_::Ptr{Cvoid},
                                                 out::Ptr{Cvoid}, kinds::Ptr{Cint}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try
        r = ld_int(rank)
        wplan_basic($T, W_R2R, rev_ints(n, r), in_, out, rev_kinds(kinds, max(r, 0)), 0, ld_flags(flags))
      catch e; @wfailed(e) end)
    end

    # --- advanced interface
    Base.@ccallable function $(fname(:plan_many_dft))(plan::Ptr{Int64}, rank::Ptr{Cint}, n::Ptr{Cint}, howmany::Ptr{Cint},
        in_::Ptr{Cvoid}, inembed::Ptr{Cint}, istride::Ptr{Cint}, idist::Ptr{Cint},
        out::Ptr{Cvoid}, onembed::Ptr{Cint}, ostride::Ptr{Cint}, odist::Ptr{Cint}, sign::Ptr{Cint}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try
        fplan_many($T, W_C2C, ld_int(rank), n, ld_int(howmany), in_, inembed, ld_int(istride), ld_int(idist),
                   out, onembed, ld_int(ostride), ld_int(odist), Ptr{Cint}(C_NULL), ld_int(sign), ld_flags(flags))
      catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_many_dft_r2c))(plan::Ptr{Int64}, rank::Ptr{Cint}, n::Ptr{Cint}, howmany::Ptr{Cint},
        in_::Ptr{Cvoid}, inembed::Ptr{Cint}, istride::Ptr{Cint}, idist::Ptr{Cint},
        out::Ptr{Cvoid}, onembed::Ptr{Cint}, ostride::Ptr{Cint}, odist::Ptr{Cint}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try
        fplan_many($T, W_R2C, ld_int(rank), n, ld_int(howmany), in_, inembed, ld_int(istride), ld_int(idist),
                   out, onembed, ld_int(ostride), ld_int(odist), Ptr{Cint}(C_NULL), -1, ld_flags(flags))
      catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_many_dft_c2r))(plan::Ptr{Int64}, rank::Ptr{Cint}, n::Ptr{Cint}, howmany::Ptr{Cint},
        in_::Ptr{Cvoid}, inembed::Ptr{Cint}, istride::Ptr{Cint}, idist::Ptr{Cint},
        out::Ptr{Cvoid}, onembed::Ptr{Cint}, ostride::Ptr{Cint}, odist::Ptr{Cint}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try
        fplan_many($T, W_C2R, ld_int(rank), n, ld_int(howmany), in_, inembed, ld_int(istride), ld_int(idist),
                   out, onembed, ld_int(ostride), ld_int(odist), Ptr{Cint}(C_NULL), 1, ld_flags(flags))
      catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_many_r2r))(plan::Ptr{Int64}, rank::Ptr{Cint}, n::Ptr{Cint}, howmany::Ptr{Cint},
        in_::Ptr{Cvoid}, inembed::Ptr{Cint}, istride::Ptr{Cint}, idist::Ptr{Cint},
        out::Ptr{Cvoid}, onembed::Ptr{Cint}, ostride::Ptr{Cint}, odist::Ptr{Cint}, kinds::Ptr{Cint}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try
        fplan_many($T, W_R2R, ld_int(rank), n, ld_int(howmany), in_, inembed, ld_int(istride), ld_int(idist),
                   out, onembed, ld_int(ostride), ld_int(odist), kinds, 0, ld_flags(flags))
      catch e; @wfailed(e) end)
    end

    # --- guru interface: iodims as (n, is, os) arrays, in the given order
    Base.@ccallable function $(fname(:plan_guru_dft))(plan::Ptr{Int64}, rank::Ptr{Cint}, n::Ptr{Cint}, is::Ptr{Cint},
        os::Ptr{Cint}, hrank::Ptr{Cint}, hn::Ptr{Cint}, his::Ptr{Cint}, hos::Ptr{Cint},
        in_::Ptr{Cvoid}, out::Ptr{Cvoid}, sign::Ptr{Cint}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try
        fplan_guru($T, W_C2C, ld_int(rank), n, is, os, ld_int(hrank), hn, his, hos, in_, out, Ptr{Cint}(C_NULL),
                   ld_int(sign), ld_flags(flags))
      catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_guru_dft_r2c))(plan::Ptr{Int64}, rank::Ptr{Cint}, n::Ptr{Cint}, is::Ptr{Cint},
        os::Ptr{Cint}, hrank::Ptr{Cint}, hn::Ptr{Cint}, his::Ptr{Cint}, hos::Ptr{Cint},
        in_::Ptr{Cvoid}, out::Ptr{Cvoid}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try
        fplan_guru($T, W_R2C, ld_int(rank), n, is, os, ld_int(hrank), hn, his, hos, in_, out, Ptr{Cint}(C_NULL),
                   -1, ld_flags(flags))
      catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_guru_dft_c2r))(plan::Ptr{Int64}, rank::Ptr{Cint}, n::Ptr{Cint}, is::Ptr{Cint},
        os::Ptr{Cint}, hrank::Ptr{Cint}, hn::Ptr{Cint}, his::Ptr{Cint}, hos::Ptr{Cint},
        in_::Ptr{Cvoid}, out::Ptr{Cvoid}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try
        fplan_guru($T, W_C2R, ld_int(rank), n, is, os, ld_int(hrank), hn, his, hos, in_, out, Ptr{Cint}(C_NULL),
                   1, ld_flags(flags))
      catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_guru_r2r))(plan::Ptr{Int64}, rank::Ptr{Cint}, n::Ptr{Cint}, is::Ptr{Cint},
        os::Ptr{Cint}, hrank::Ptr{Cint}, hn::Ptr{Cint}, his::Ptr{Cint}, hos::Ptr{Cint},
        in_::Ptr{Cvoid}, out::Ptr{Cvoid}, kinds::Ptr{Cint}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try
        fplan_guru($T, W_R2R, ld_int(rank), n, is, os, ld_int(hrank), hn, his, hos, in_, out, kinds, 0, ld_flags(flags))
      catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_guru_split_dft))(plan::Ptr{Int64}, rank::Ptr{Cint}, n::Ptr{Cint}, is::Ptr{Cint},
        os::Ptr{Cint}, hrank::Ptr{Cint}, hn::Ptr{Cint}, his::Ptr{Cint}, hos::Ptr{Cint},
        ri::Ptr{Cvoid}, ii::Ptr{Cvoid}, ro::Ptr{Cvoid}, io::Ptr{Cvoid}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try
        fplan_guru($T, W_C2C, ld_int(rank), n, is, os, ld_int(hrank), hn, his, hos, ri, ro, Ptr{Cint}(C_NULL),
                   -1, ld_flags(flags); split = true, pin2 = ii, pout2 = io)
      catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_guru_split_dft_r2c))(plan::Ptr{Int64}, rank::Ptr{Cint}, n::Ptr{Cint}, is::Ptr{Cint},
        os::Ptr{Cint}, hrank::Ptr{Cint}, hn::Ptr{Cint}, his::Ptr{Cint}, hos::Ptr{Cint},
        in_::Ptr{Cvoid}, ro::Ptr{Cvoid}, io::Ptr{Cvoid}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try
        fplan_guru($T, W_R2C, ld_int(rank), n, is, os, ld_int(hrank), hn, his, hos, in_, ro, Ptr{Cint}(C_NULL),
                   -1, ld_flags(flags); split = true, pout2 = io)
      catch e; @wfailed(e) end)
    end
    Base.@ccallable function $(fname(:plan_guru_split_dft_c2r))(plan::Ptr{Int64}, rank::Ptr{Cint}, n::Ptr{Cint}, is::Ptr{Cint},
        os::Ptr{Cint}, hrank::Ptr{Cint}, hn::Ptr{Cint}, his::Ptr{Cint}, hos::Ptr{Cint},
        ri::Ptr{Cvoid}, ii::Ptr{Cvoid}, out::Ptr{Cvoid}, flags::Ptr{Cint})::Cvoid
      st_plan(plan, try
        fplan_guru($T, W_C2R, ld_int(rank), n, is, os, ld_int(hrank), hn, his, hos, ri, out, Ptr{Cint}(C_NULL),
                   1, ld_flags(flags); split = true, pin2 = ii)
      catch e; @wfailed(e) end)
    end

    # --- execution
    Base.@ccallable function $(fname(:execute))(plan::Ptr{Int64})::Cvoid
      try wexecute($T, ld_plan(plan), -1, false, C_NULL, C_NULL, C_NULL, C_NULL, false) catch; set_error!(W_EXEC_FAILED) end
      nothing
    end
    Base.@ccallable function $(fname(:execute_dft))(plan::Ptr{Int64}, in_::Ptr{Cvoid}, out::Ptr{Cvoid})::Cvoid
      try wexecute($T, ld_plan(plan), W_C2C, false, in_, C_NULL, out, C_NULL, true) catch; set_error!(W_EXEC_FAILED) end
      nothing
    end
    Base.@ccallable function $(fname(:execute_dft_r2c))(plan::Ptr{Int64}, in_::Ptr{Cvoid}, out::Ptr{Cvoid})::Cvoid
      try wexecute($T, ld_plan(plan), W_R2C, false, in_, C_NULL, out, C_NULL, true) catch; set_error!(W_EXEC_FAILED) end
      nothing
    end
    Base.@ccallable function $(fname(:execute_dft_c2r))(plan::Ptr{Int64}, in_::Ptr{Cvoid}, out::Ptr{Cvoid})::Cvoid
      try wexecute($T, ld_plan(plan), W_C2R, false, in_, C_NULL, out, C_NULL, true) catch; set_error!(W_EXEC_FAILED) end
      nothing
    end
    Base.@ccallable function $(fname(:execute_r2r))(plan::Ptr{Int64}, in_::Ptr{Cvoid}, out::Ptr{Cvoid})::Cvoid
      try wexecute($T, ld_plan(plan), W_R2R, false, in_, C_NULL, out, C_NULL, true) catch; set_error!(W_EXEC_FAILED) end
      nothing
    end
    Base.@ccallable function $(fname(:execute_split_dft))(plan::Ptr{Int64}, ri::Ptr{Cvoid}, ii::Ptr{Cvoid}, ro::Ptr{Cvoid},
                                                          io::Ptr{Cvoid})::Cvoid
      try wexecute($T, ld_plan(plan), W_C2C, true, ri, ii, ro, io, true) catch; set_error!(W_EXEC_FAILED) end
      nothing
    end
    Base.@ccallable function $(fname(:execute_split_dft_r2c))(plan::Ptr{Int64}, in_::Ptr{Cvoid}, ro::Ptr{Cvoid},
                                                              io::Ptr{Cvoid})::Cvoid
      try wexecute($T, ld_plan(plan), W_R2C, true, in_, C_NULL, ro, io, true) catch; set_error!(W_EXEC_FAILED) end
      nothing
    end
    Base.@ccallable function $(fname(:execute_split_dft_c2r))(plan::Ptr{Int64}, ri::Ptr{Cvoid}, ii::Ptr{Cvoid},
                                                              out::Ptr{Cvoid})::Cvoid
      try wexecute($T, ld_plan(plan), W_C2R, true, ri, ii, out, C_NULL, true) catch; set_error!(W_EXEC_FAILED) end
      nothing
    end

    # --- plans
    Base.@ccallable function $(fname(:destroy_plan))(plan::Ptr{Int64})::Cvoid
      p = ld_plan(plan)
      p == C_NULL || wdestroy($T, p, false) || set_error!("FFTW interface: invalid plan")
      nothing
    end
    Base.@ccallable function $(fname(:copy_plan))(newplan::Ptr{Int64}, plan::Ptr{Int64})::Cvoid
      p = ld_plan(plan)
      st_plan(newplan, wdestroy($T, p, true) ? p : (set_error!("FFTW interface: invalid plan"); Ptr{Cvoid}(C_NULL)))
    end
    Base.@ccallable function $(fname(:cleanup))()::Cvoid
      FFTL.clear_cache!(); nothing
    end
    Base.@ccallable function $(fname(:set_timelimit))(t::Ptr{Cdouble})::Cvoid
      nothing
    end
    Base.@ccallable function $(fname(:cost))(c::Ptr{Cdouble}, plan::Ptr{Int64})::Cvoid
      _, ops, _ = winfo($T, ld_plan(plan))
      unsafe_store!(c, ops[1] + ops[2]); nothing
    end
    Base.@ccallable function $(fname(:flops))(plan::Ptr{Int64}, add::Ptr{Cdouble}, mul::Ptr{Cdouble}, fma::Ptr{Cdouble})::Cvoid
      _, ops, _ = winfo($T, ld_plan(plan))
      add == C_NULL || unsafe_store!(add, ops[1])
      mul == C_NULL || unsafe_store!(mul, ops[2])
      fma == C_NULL || unsafe_store!(fma, 0.0)
      nothing
    end
    Base.@ccallable function $(fname(:print_plan))(plan::Ptr{Int64})::Cvoid
      s, _, ok = winfo($T, ld_plan(plan))
      ok ? c_fputs(s, c_stdout()) : set_error!("FFTW interface: invalid plan")
      nothing
    end

    # --- wisdom
    Base.@ccallable function $(fname(:export_wisdom))(write_char::Ptr{Cvoid}, data::Ptr{Cvoid})::Cvoid
      fexport_callback(write_char, data)
    end
    Base.@ccallable function $(fname(:import_wisdom))(isuccess::Ptr{Cint}, read_char::Ptr{Cvoid}, data::Ptr{Cvoid})::Cvoid
      unsafe_store!(isuccess, Cint(1)); nothing
    end
    Base.@ccallable function $(fname(:import_system_wisdom))(isuccess::Ptr{Cint})::Cvoid
      unsafe_store!(isuccess, Cint(0)); nothing
    end
    Base.@ccallable function $(fname(:forget_wisdom))()::Cvoid
      nothing
    end

    # --- threads
    Base.@ccallable function $(fname(:init_threads))(iret::Ptr{Cint})::Cvoid
      unsafe_store!(iret, Cint(1)); nothing
    end
    Base.@ccallable function $(fname(:plan_with_nthreads))(n::Ptr{Cint})::Cvoid
      wthreads($T)[] = max(1, ld_int(n)); nothing
    end
    Base.@ccallable function $(fname(:planner_nthreads))(iret::Ptr{Cint})::Cvoid
      unsafe_store!(iret, Cint(wthreads($T)[])); nothing
    end
    Base.@ccallable function $(fname(:cleanup_threads))()::Cvoid
      FFTL.clear_cache!(); nothing
    end
  end
end
