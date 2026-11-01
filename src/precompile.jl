# Precompile workload (variant BROAD) -- draft, copied into src/precompile.jl for measurement.
# The benchmark shapes plus lengths that add the radix families 3, 5, 6, 8, 9, 10, 12, Bluestein,
# run-time plans, real transforms, passes, and one four-step (threaded long line) case.

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
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
    # Do not serialise the twiddle/filter tables built while compiling: the image should contain
    # code, not the runtime cache (a 2^20 chain alone is tens of MB).
    clear_cache!()
end
