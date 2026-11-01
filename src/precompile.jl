# Precompile workload: build and apply the plans whose generated kernels dominate the first
# plan and first transform in a fresh process. The cost is almost all compilation of the
# codelets and of the code that selects them, so the workload runs small arrays; one length
# per radix family (4, 10/5, 12/6, 9/3, 8, 16), one Bluestein length, small 2D/3D arrays,
# run-time plans, real transforms and passes, for both element types.

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    for T in (Float64, Float32)
        xs = [randn(Complex{T}, n) for n in (1024, 1000, 360, 243, 128, 64, 1009)]
        x2 = randn(Complex{T}, 64, 64)
        x3 = randn(Complex{T}, 16, 16, 16)
        xr1 = randn(T, 1024)
        xr2 = randn(T, 64, 64)
        zp = randn(Complex{T}, 64, 48)
        v1 = view(zp, 5:60, 3:45)
        v2 = view(zp, 5:60, :)
        @compile_workload begin
            for x in (xs..., x2, x3)
                p = plan_fft(x; threads = 1)
                mul!(similar(x), p, x)
                p = plan_bfft(x; threads = 1)
                mul!(similar(x), p, x)
                p = plan_ifft(x; threads = 1)
                mul!(similar(x), p, x)
            end
            # run-time plans: one smooth 1D, one Bluestein, one 2D and one 3D
            for (sz, reg) in (([1024], [1]), ([1009], [1]), ([64, 64], [1, 2]), ([16, 16, 16], [1, 2, 3]))
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
        end
    end
    # Do not serialise the twiddle/filter tables built while compiling: the image should contain
    # code, not the runtime cache (a 2^20 chain alone is tens of MB).
    clear_cache!()
end
