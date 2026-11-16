# Precompile workload, selected with the `precompile_workload` preference of FFTL:
#
#   julia> using Preferences, FFTL
#   julia> Preferences.set_preferences!(FFTL, "precompile_workload" => "broad"; force = true)
#
# The next load recompiles FFTL with the new workload. Values:
#
#   "lean"  (default) 1024/1000/1009 in 1D, 16×16 and 8×8×8, both element types, all three
#                      transforms, RPlan, plan_rfft and plan_pass. It covers the common
#                      codelets (radices 4, 5, 8, 9, 10 and 16, plus resampling) and the
#                      plan types of both the default and the run-time plans.
#   "broad"            the measured shapes 10^5, 2^20, 256×256, 64×64×64 plus the radix
#                      12/6/3 lengths (360, 243), 128, and a four-step (threads = 4) case.
#   "none"             no workload at all (for CI or when the image size is more important
#                      than the first plan and first transform).
#
# The workload builds and applies plans: the expensive generated methods are only compiled
# when a transform runs, so constructing a plan is not enough. Plans are explicitly
# single-threaded so that the result does not depend on the thread count of the precompiling
# process. `FFTL.KCACHE` is cleared at the end of the setup so that the package image
# contains code but not the runtime twiddle/filter tables (a 2^20 chain alone is tens of MB).

using Preferences: @load_preference
using PrecompileTools: @setup_workload, @compile_workload

"""
    FFTL.PRECOMPILE_WORKLOAD

The precompile workload selected when FFTL was built: `"lean"` (default), `"broad"` or
`"none"`. It comes from the `precompile_workload` preference:

```julia
using Preferences, FFTL
Preferences.set_preferences!(FFTL, "precompile_workload" => "broad"; force = true)
```

The next load recompiles FFTL with the selected workload.
"""
const PRECOMPILE_WORKLOAD = @load_preference("precompile_workload", "lean")

# Plans built and applied by the workload for element type T: complex transforms of the
# shapes (forward, backward, inverse), run-time plans, real transforms, passes on views.
function precompile_workload(::Type{T}, shapes, rshapes, rplans, zp; fourstep = false) where {T}
  for sz in shapes
    x = randn(Complex{T}, sz...)
    for plan in (plan_fft, plan_bfft, plan_ifft)
      p = plan(x; threads = 1)
      mul!(similar(x), p, x)
    end
  end
  for (sz, reg) in rplans
    x = randn(Complex{T}, sz...)
    p = RPlan{T}(sz, reg, -1; threads = 1)
    mul!(similar(x), p, x)
  end
  for sz in rshapes
    x = randn(T, sz...)
    plan_rfft(x; threads = 1) * x
  end
  z = randn(Complex{T}, zp...)
  for (v, d) in ((view(z, 2:zp[1]-1, 2:zp[2]-1), 1), (view(z, 2:zp[1]-1, :), 2))
    apply!(plan_pass(v, d), v)
  end
  if fourstep   # one long line on several threads
    x = randn(Complex{T}, 2^14)
    mul!(similar(x), plan_fft(x; threads = 4), x)
  end
  nothing
end

if PRECOMPILE_WORKLOAD != "none"
  @setup_workload begin
    broad = PRECOMPILE_WORKLOAD == "broad"
    shapes = broad ?
      [(1024,), (100_000,), (2^20,), (1000,), (360,), (243,), (128,), (64,), (1009,), (256, 256), (64, 64, 64)] :
      [(1024,), (1000,), (1009,), (16, 16), (8, 8, 8)]
    rshapes = broad ? [(1024,), (64, 64)] : [(1024,)]
    rplans = broad ? [([1024], [1]), ([1009], [1]), ([256, 256], [1, 2]), ([64, 64, 64], [1, 2, 3])] :
                     [([1024], [1]), ([1009], [1]), ([16, 16], [1, 2]), ([8, 8, 8], [1, 2, 3])]
    zp = broad ? (64, 48) : (16, 16)
    @compile_workload begin
      for T in (Float64, Float32)
        precompile_workload(T, shapes, rshapes, rplans, zp; fourstep = broad)
      end
    end
    # Do not serialise the twiddle/filter tables built while compiling: the image should
    # contain code, not the runtime cache.
    clear_cache!()
  end
end
