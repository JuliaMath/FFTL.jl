# Radix sequences chosen by the planner, per platform: test/data/radices-<arch>.csv, one line per
# (length, stride) with the radices in order. The test compares the planner with the file, so a
# change of the rules in factor.jl shows up as a diff of this file. Regenerate with
#   julia --project=. test/radix_table.jl
# after a change that was checked with the dense sweep (bench/x86/sweep_small.jl) on that platform.
using FFTL

const RADIX_TABLE_STRIDES = (1, 1024)
const RADIX_TABLE_LARGE = (2048, 4096, 8192, 16384, 32768, 65536, 100000, 120000, 131072, 262144,
                           786432, 1048576)

# Lengths with prime factors up to MAX_RADIX_PRIME (the lengths a chain handles), from 2 to 1024
# and the large ones above.
radix_table_lengths() =
  vcat(filter(n -> FFTL.factorize(n)[2] == 1, 2:1024), collect(RADIX_TABLE_LARGE))

radix_table_path(arch = string(Sys.ARCH)) = joinpath(@__DIR__, "data", "radices-$arch.csv")

function radix_table_lines()
  lines = String[]
  for n in radix_table_lengths(), S in RADIX_TABLE_STRIDES
    push!(lines, string(n, ",", S, ",", join(FFTL.radices(n, S), " ")))
  end
  lines
end

if abspath(PROGRAM_FILE) == @__FILE__
  write(radix_table_path(), join(radix_table_lines(), "\n") * "\n")
  println("wrote ", radix_table_path())
end
