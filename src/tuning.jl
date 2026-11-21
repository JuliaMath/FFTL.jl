# Thresholds of the planner and of the threaded passes, in one place. They are read when a plan is
# made or when a pass is started (never inside a stage), so changing them costs nothing per
# element. The values were measured on an Apple M1 Max and on an AMD Zen 3; the comments at the
# uses give the evidence for each.

"""
    Tuning

Size thresholds of FFTL (see `TUNING`):
- `thread_min_work`: complex values of a pass below which it runs on one thread
- `block_budget`: complex values of one lane-block buffer on one thread (threaded long lines use
  `clamp(prevpow2(L2 ÷ 32), 2^12, block_budget)`)
- `fourstep_min`, `fourstep_min_bytes`, `fourstep_min_threads`: a single smooth line uses the
  four-step algorithm on several threads from this length, from this many bytes and from this many
  threads (all three must hold)
- `factor_twiddles`: bytes above which the four-step uses factor twiddle tables
- `resample_mt_min`: length from which a length with a large prime factor (resampling) runs on
  several threads
- `real_fourstep_bytes`: bytes of a real line from which it runs as a four-step on several threads
- `real_resample_bytes`: bytes of a real line from which a non-smooth one runs on several threads
- `plane_budget`: complex values of a plane (dimensions 1 and 2) that may be fused
- `fourstep_nt_bytes`: bytes of a four-step from which phase 2 writes its output rows from a buffer
  with non-temporal stores (below: the last stage writes y); with 64-byte vectors only
- `fourstep_row_bytes`: least bytes per row of a lane block of a threaded four-step
"""
struct Tuning
  thread_min_work::Int
  block_budget::Int
  fourstep_min::Int
  fourstep_min_bytes::Int
  fourstep_min_threads::Int
  factor_twiddles::Int
  resample_mt_min::Int
  real_fourstep_bytes::Int
  real_resample_bytes::Int
  plane_budget::Int
  fourstep_nt_bytes::Int
  fourstep_row_bytes::Int
end

# fourstep_min*: measured per architecture, see `use_fourstep` in src/threads.jl.
const DEFAULT_TUNING = Sys.ARCH === :aarch64 ?
  Tuning(1 << 15, 1 << 16, 0, 128 << 10, 2, 1 << 23, 1 << 14, 1 << 18, 1 << 17, 1 << 17, 8 << 20, 128) :
  Tuning(1 << 15, 1 << 16, 1 << 15, 0, 6, 1 << 23, 1 << 14, 1 << 18, 1 << 17, 1 << 17, 8 << 20, 128)

# Read at plan construction, except thread_min_work, block_budget, plane_budget and the fourstep_*
# fields, which a pass reads when it starts (fourstep_row_bytes also sizes a four-step's buffers when
# the plan is made). Assigning `FFTL.TUNING[] = …` changes the settings for plans made afterwards
# and for passes started afterwards.
const TUNING = Ref(DEFAULT_TUNING)
