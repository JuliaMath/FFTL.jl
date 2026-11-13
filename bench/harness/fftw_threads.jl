# FFTW threading without FFTW.jl's Julia callback.
#
# With Julia started with more than one thread, FFTW.jl registers a callback
# (fftw_threads_set_callback) so that FFTW's parallel loops run as Julia tasks. In the campaign
# FFTW instead runs its own worker threads: fftw_pthreads!() resets the callback to NULL for
# both precisions (FFTW then uses its built-in threads again, which needs a libfftw3 built with
# threads, as FFTW_jll and bench/fftw/build_native.sh are). Call it before the first plan.
import FFTW

function fftw_pthreads!()
  FFTW.set_num_threads(1)   # runs FFTW.jl's lazy init (fftw_init_threads, callback) first
  for (lib, f) in ((FFTW.FFTW_jll.libfftw3_path, :fftw_threads_set_callback),
                   (FFTW.FFTW_jll.libfftw3f_path, :fftwf_threads_set_callback))
    ccall(Libc.Libdl.dlsym(Libc.Libdl.dlopen(lib), f), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), C_NULL, C_NULL)
  end
end

# FFTW's parallel loops as Julia tasks like FFTW.jl's callback, but sticky: every task, including
# the one that waits for the work items, stays on the thread it started on. A task that waits
# while FFTW's C frames are on its stack is then never resumed on another thread.
function sticky_loop(f::Ptr{Cvoid}, fdata::Ptr{Cvoid}, elsize::Csize_t, num::Cint, ::Ptr{Cvoid})
  current_task().sticky = true
  ts = map(0:num-1) do i
    t = Task(() -> ccall(f, Ptr{Cvoid}, (Ptr{Cvoid},), fdata + elsize * i)); t.sticky = true
    ccall(:jl_set_task_tid, Cint, (Any, Cint), t, i % Threads.nthreads()); schedule(t); t
  end
  foreach(wait, ts)
end
function fftw_sticky!()
  FFTW.set_num_threads(1)
  cb = @cfunction(sticky_loop, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint, Ptr{Cvoid}))
  for (lib, f) in ((FFTW.FFTW_jll.libfftw3_path, :fftw_threads_set_callback),
                   (FFTW.FFTW_jll.libfftw3f_path, :fftwf_threads_set_callback))
    ccall(Libc.Libdl.dlsym(Libc.Libdl.dlopen(lib), f), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), cb, C_NULL)
  end
end
# CAMPAIGN_FFTW_THREADS: pthreads, sticky or callback (FFTW.jl's default)
function fftw_threads_mode!(mode)
  mode == "pthreads" ? fftw_pthreads!() : mode == "sticky" ? fftw_sticky!() :
  mode == "callback" ? nothing : error("unknown FFTW thread mode $mode")
end

os_threads() = Sys.islinux() ? length(readdir("/proc/self/task")) : -1
