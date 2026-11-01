# Counts the trim-verifier errors per entry point: an error block starts with
# "Error #i: unresolved call ..." and the entry function appears in its stack; the block ends
# at the next "Error #" line. Errors without an entry point in the stack are reported
# separately. Run: julia --project=. count.jl build/trimcheck.log

const CASES = ("fftl_trim_1d_f64", "fftl_trim_1d_f32", "fftl_trim_2d_f64", "fftl_trim_rfft_f64",
               "fftl_trim_r2r_f64", "fftl_trim_plan_f64", "fftl_trim_exec_f64")

function count_errors(path)
  lines = readlines(path)
  counts = Dict(name => 0 for name in CASES)
  unattributed = 0
  total = 0
  i = 1
  while i <= length(lines)
    if startswith(lines[i], "Error #") && occursin("unresolved", lines[i])
      total += 1
      case = ""
      j = i + 1
      while j <= length(lines) && !startswith(lines[j], "Error #")
        if isempty(case)
          k = findfirst(name -> occursin(name, lines[j]), CASES)
          k !== nothing && (case = CASES[k])
        end
        j += 1
      end
      if isempty(case)
        unattributed += 1
      else
        counts[case] += 1
      end
      i = j
    else
      i += 1
    end
  end
  counts, unattributed, total
end

using Printf
path = isempty(ARGS) ? "build/trimcheck.log" : ARGS[1]
counts, unattributed, total = count_errors(path)
for name in CASES
  @printf("  %-24s %4d\n", name, counts[name])
end
@printf("  %-24s %4d\n", "(unattributed)", unattributed)
@printf("  %-24s %4d\n", "total", total)
total == 0 && println("  trim clean: no unresolved calls")
