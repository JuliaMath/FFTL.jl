# Cache sizes of the CPU the package runs on. Lane blocks are sized from them (threads.jl). They are
# read from the operating system when the package is loaded (macOS: sysctl of the performance cores,
# Linux: sysfs of cpu0, Windows: GetLogicalProcessorInformationEx for logical processor 0). Only two
# numbers are needed, so the queries are made directly instead of through a topology library.
# Where a query fails, defaults stand in (DEFAULT_CACHES).

struct CPUCaches
  l1::Int        # L1 data cache per core, bytes
  l2::Int        # L2 per core, bytes (a shared L2 divided by the cores sharing it)
  llc::Int       # last-level cache of one core cluster, bytes (L3, or the shared L2 on Apple)
  source::Symbol # :queried, :default or :user
end

# Defaults where the query fails. The L2 sets the block size of threaded passes (threads.jl): a
# smaller value gives smaller blocks, which fit more caches, so the x86 default is the smallest
# per-core L2 of current x86 cores (256 KiB). The LLC sets where a single line takes the serial
# four-step (only with 64-byte vectors, threads.jl); a larger value keeps the chain, the safer side,
# so it stays at the 32 MiB of the measured Zen 3 / Zen 4 core clusters.
const DEFAULT_CACHES = Sys.ARCH === :aarch64 ?
  CPUCaches(128 << 10, 3 << 20, 12 << 20, :default) :   # M1 Max: 12 MB L2 per 4 P-cores
  CPUCaches(32 << 10, 256 << 10, 32 << 20, :default)

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

# Windows: GetLogicalProcessorInformationEx(RelationAll) fills a list of variable-length records
# SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: relationship (UInt32) at byte 0, record size (UInt32) at 4.
#   processor core (0): group count (UInt16) at 30, group affinities from 32
#   cache (2): level (UInt8) at 8, size (UInt32) at 12, type (UInt32; 1 instruction, 3 trace) at 16,
#              group count (UInt16) at 38, group affinities from 40
# A group affinity is 16 bytes: processor mask (UInt64) at 0, group (UInt16) at 8. Windows before 11
# leaves the group count 0 and has one affinity. Layout for 64-bit Windows only.
winload(::Type{T}, b, off) where {T} = GC.@preserve b unsafe_load(Ptr{T}(pointer(b) + off))

# Logical processors in the group affinities at off, and whether processor 0 of group 0 is one of them.
function win_affinity(b, off, count)
  n = 0; has0 = false
  for k in 0:max(1, Int(count))-1
    mask = winload(UInt64, b, off + 16k)
    n += count_ones(mask)
    has0 |= winload(UInt16, b, off + 16k + 8) == 0 && isodd(mask)
  end
  n, has0
end

function parse_processor_info(b::Vector{UInt8})
  smt = 1; l1 = l2 = llc = 0; l2cpus = 1
  off = 0
  while off + 8 <= length(b)
    rel = winload(UInt32, b, off); sz = Int(winload(UInt32, b, off + 4))
    (sz >= 8 && off + sz <= length(b)) || break
    if rel == 0 && sz >= 48
      n, has0 = win_affinity(b, off + 32, winload(UInt16, b, off + 30))
      has0 && (smt = max(1, n))
    elseif rel == 2 && sz >= 56
      n, has0 = win_affinity(b, off + 40, winload(UInt16, b, off + 38))
      typ = winload(UInt32, b, off + 16)
      if has0 && (typ == 0 || typ == 2)
        level = winload(UInt8, b, off + 8); size = Int(winload(UInt32, b, off + 12))
        level == 1 && (l1 = size)
        level == 2 && (l2 = size; l2cpus = n)
        level >= 2 && (llc = max(llc, size))
      end
    end
    off += sz
  end
  l1 > 0 && l2 > 0 || return nothing
  CPUCaches(l1, l2 ÷ max(1, l2cpus ÷ smt), llc, :queried)
end

function query_caches_windows()
  Sys.WORD_SIZE == 64 || return nothing
  len = Ref{UInt32}(0)
  ccall((:GetLogicalProcessorInformationEx, "kernel32"), stdcall, Cint, (UInt32, Ptr{UInt8}, Ref{UInt32}),
        0xffff, C_NULL, len)
  len[] == 0 && return nothing
  b = Vector{UInt8}(undef, len[])
  ok = ccall((:GetLogicalProcessorInformationEx, "kernel32"), stdcall, Cint, (UInt32, Ptr{UInt8}, Ref{UInt32}),
             0xffff, b, len) != 0
  ok ? parse_processor_info(resize!(b, len[])) : nothing
end

function query_caches()
  c = try
    Sys.isapple() ? query_caches_macos() : Sys.islinux() ? query_caches_linux() :
      Sys.iswindows() ? query_caches_windows() : nothing
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
