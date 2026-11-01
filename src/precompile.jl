# Precompile workload, selected with the `precompile_workload` preference of FFTL:
#
#   julia> using Preferences, FFTL
#   julia> Preferences.set_preferences!(FFTL, "precompile_workload" => "broad"; force = true)
#
# The next load recompiles FFTL with the new workload. Values:
#
#   "lean"  (default) 1024/1000/1009 in 1D, 16×16 and 8×8×8, both element types, all three
#                      transforms, RPlan, plan_rfft and plan_pass. It covers the common
#                      codelets (radices 4, 5, 8, 9, 10 and 16, plus Bluestein) and the JIT
#                      paths for both the compile-time and the run-time plans.
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

if PRECOMPILE_WORKLOAD != "none"
    @setup_workload begin
        if PRECOMPILE_WORKLOAD == "broad"
            for T in (Float64, Float32)
                # 1D: benchmark lengths plus lengths that add the 3/5 and radix-8 codelets
                xs = [randn(Complex{T}, n) for n in (1024, 100_000, 2^20, 1000, 360, 243, 128, 64, 1009)]
                x2 = randn(Complex{T}, 256, 256)
                x3 = randn(Complex{T}, 64, 64, 64)
                xr1 = randn(T, 1024)
                xr2 = randn(T, 64, 64)
                zp = randn(Complex{T}, 64, 48)
                v1 = view(zp, 5:60, 3:45)
                v2 = view(zp, 5:60, :)
                zl = randn(Complex{T}, 2^14)
                @compile_workload begin
                    for x in (xs..., x2, x3)
                        p = plan_fft(x; threads = 1)
                        mul!(similar(x), p, x)
                        p = plan_bfft(x; threads = 1)
                        mul!(similar(x), p, x)
                        p = plan_ifft(x; threads = 1)
                        mul!(similar(x), p, x)
                    end
                    for (sz, reg) in (([1024], [1]), ([1009], [1]), ([256, 256], [1, 2]), ([64, 64, 64], [1, 2, 3]))
                        x = randn(Complex{T}, sz...)
                        p = RPlan{T}(sz, reg, -1; threads = 1)
                        mul!(similar(x), p, x)
                    end
                    for x in (xr1, xr2)
                        p = plan_rfft(x; threads = 1)
                        p * x
                    end
                    for (v, d) in ((v1, 1), (v2, 2))
                        p = plan_pass(v, d)
                        apply!(p, v)
                    end
                    # four-step for one long line on several threads
                    p = plan_fft(zl; threads = 4)
                    mul!(similar(zl), p, zl)
                end
            end
        else
            for T in (Float64, Float32)
                xs = [randn(Complex{T}, n) for n in (1024, 1000, 1009)]
                x2 = randn(Complex{T}, 16, 16)
                x3 = randn(Complex{T}, 8, 8, 8)
                xr1 = randn(T, 1024)
                zp = randn(Complex{T}, 16, 16)
                v1 = view(zp, 2:15, 2:15)
                v2 = view(zp, 2:15, :)
                @compile_workload begin
                    for x in (xs..., x2, x3)
                        p = plan_fft(x; threads = 1)
                        mul!(similar(x), p, x)
                        p = plan_bfft(x; threads = 1)
                        mul!(similar(x), p, x)
                        p = plan_ifft(x; threads = 1)
                        mul!(similar(x), p, x)
                    end
                    for (sz, reg) in (([1024], [1]), ([1009], [1]), ([16, 16], [1, 2]), ([8, 8, 8], [1, 2, 3]))
                        x = randn(Complex{T}, sz...)
                        p = RPlan{T}(sz, reg, -1; threads = 1)
                        mul!(similar(x), p, x)
                    end
                    p = plan_rfft(xr1; threads = 1)
                    p * xr1
                    for (v, d) in ((v1, 1), (v2, 2))
                        p = plan_pass(v, d)
                        apply!(p, v)
                    end
                end
            end
        end
        # Do not serialise the twiddle/filter tables built while compiling: the image should
        # contain code, not the runtime cache.
        clear_cache!()
    end
end
