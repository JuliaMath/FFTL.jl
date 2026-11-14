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
- `fourstep_min`: length from which a single line uses the four-step algorithm on several threads
- `factor_twiddles`: bytes above which the four-step uses factor twiddle tables
- `resample_mt_min`: length from which a length with a large prime factor (resampling) runs on
  several threads
- `real_fourstep_bytes`: bytes of a real line from which it runs as a four-step on several threads
- `real_resample_bytes`: bytes of a real line from which a non-smooth one runs on several threads
- `plane_budget`: complex values of a plane (dimensions 1 and 2) that may be fused
"""
struct Tuning
  thread_min_work::Int
  block_budget::Int
  fourstep_min::Int
  factor_twiddles::Int
  resample_mt_min::Int
  real_fourstep_bytes::Int
  real_resample_bytes::Int
  plane_budget::Int
end

const DEFAULT_TUNING = Tuning(1 << 15, 1 << 16, 1 << 13, 1 << 23, 1 << 14, 1 << 18, 1 << 17, 1 << 17)

# Read at plan construction, except thread_min_work, block_budget and plane_budget, which a pass
# reads when it starts. Assigning `FFTL.TUNING[] = …` changes the settings for plans made afterwards
# and for passes started afterwards.
const TUNING = Ref(DEFAULT_TUNING)
