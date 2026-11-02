# Cache sizes of the CPU the package runs on. Lane blocks are sized from them (threads.jl). They are read when the package is loaded
# (macOS: sysctl of the performance cores, Linux: sysfs of cpu0); where that fails, defaults of
# the two measured machines stand in: Apple M1 Max on aarch64, AMD Zen 3 elsewhere.

struct CPUCaches
  l1::Int        # L1 data cache per core, bytes
  l2::Int        # L2 per core, bytes (a shared L2 divided by the cores sharing it)
  llc::Int       # last-level cache of one core cluster, bytes (L3, or the shared L2 on Apple)
  source::Symbol # :queried, :default or :user
end

const DEFAULT_CACHES = Sys.ARCH === :aarch64 ?
  CPUCaches(128 << 10, 3 << 20, 12 << 20, :default) :   # M1 Max: 12 MB L2 per 4 P-cores
  CPUCaches(32 << 10, 512 << 10, 32 << 20, :default)    # Zen 3: 32 MB L3 per 8 cores

const CACHES = Ref(DEFAULT_CACHES)

function sysctl_int(name)
  v = Ref{Int64}(0); sz = Ref{Csize_t}(sizeof(Int64))
  ok = ccall(:sysctlbyname, Cint, (Cstring, Ptr{Int64}, Ptr{Csize_t}, Ptr{Cvoid}, Csize_t),
             name, v, sz, C_NULL, 0) == 0
  ok ? Int(v[]) : 0
end

function query_caches_macos()
  l1 = sysctl_int("hw.perflevel0.l1dcachesize")
  l2 = sysctl_int("hw.perflevel0.l2cachesize")
  per = max(1, sysctl_int("hw.perflevel0.cpusperl2"))
  l3 = sysctl_int("hw.perflevel0.l3cachesize")
  l1 > 0 && l2 > 0 || return nothing
  CPUCaches(l1, l2 ÷ per, l3 > 0 ? l3 : l2, :queried)
end

# "512K", "32768K", "1M" -> bytes
function parse_size(s)
  s = strip(s)
  m = match(r"^(\d+)\s*([KMG]?)$", s)
  m === nothing && return 0
  parse(Int, m[1]) << (m[2] == "K" ? 10 : m[2] == "M" ? 20 : m[2] == "G" ? 30 : 0)
end

# "0-7,128-135" -> 16
count_cpus(s) = sum((r = split(p, '-'); length(r) == 1 ? 1 : parse(Int, r[2]) - parse(Int, r[1]) + 1)
                    for p in split(strip(s), ',') if !isempty(p); init = 0)

function query_caches_linux()
  base = "/sys/devices/system/cpu/cpu0"
  isdir(joinpath(base, "cache")) || return nothing
  smt = max(1, count_cpus(read(joinpath(base, "topology/thread_siblings_list"), String)))
  l1 = l2 = llc = 0
  for d in readdir(joinpath(base, "cache"); join = true)
    startswith(basename(d), "index") || continue
    rd(f) = strip(read(joinpath(d, f), String))
    rd("type") == "Instruction" && continue
    level = parse(Int, rd("level")); size = parse_size(rd("size"))
    cores = max(1, count_cpus(rd("shared_cpu_list")) ÷ smt)
    level == 1 && (l1 = size)
    level == 2 && (l2 = size ÷ cores)
    level >= 2 && (llc = max(llc, size))
  end
  l1 > 0 && l2 > 0 || return nothing
  CPUCaches(l1, l2, llc, :queried)
end

function query_caches()
  c = try
    Sys.isapple() ? query_caches_macos() : Sys.islinux() ? query_caches_linux() : nothing
  catch
    nothing
  end
  something(c, DEFAULT_CACHES)
end

"""
    FFTL.cpu_caches()

Cache sizes the CPU plans are sized from: L1 data and L2 per core, last-level cache per core
cluster, in bytes, and where they come from (`:queried`, `:default`, `:user`).
"""
cpu_caches() = CACHES[]

"""
    FFTL.set_cpu_caches!(; l1, l2, llc)

Override the cache sizes (bytes) that plans are sized from, e.g. when the query at load time
fails or a machine reports misleading values. Affects plans made
afterwards. `FFTL.set_cpu_caches!()` restores the queried values.
"""
function set_cpu_caches!(; kw...)
  q = query_caches()
  CACHES[] = isempty(kw) ? q :
    CPUCaches(get(kw, :l1, q.l1), get(kw, :l2, q.l2), get(kw, :llc, q.llc), :user)
  CACHES[]
end

prevpow2(x) = x < 1 ? 0 : prevpow(2, x)

function __init__()
  CACHES[] = query_caches()
end
