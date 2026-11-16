# Which libfftw3 the measurement loads. FFTW.jl takes it from FFTW_jll, whose library paths are
# preferences read when FFTW_jll is compiled. A machine profile names its native build:
#
#   fftw_lib  = "/path/to/fftw-native/lib/libfftw3.so"
#   fftwf_lib = "/path/to/fftw-native/lib/libfftw3f.so"
#
# use_native_fftw! writes them into the LocalPreferences.toml of the active project before FFTW is
# loaded (a changed preference recompiles FFTW_jll), and fftw_provenance reports the library that
# was actually loaded, its version and its compiler flags for the .meta. A CPU profile without
# fftw_lib stops the run unless it says fftw = "jll": the campaign compares against native builds.

using TOML, Libdl

function use_native_fftw!(profile)
  haskey(profile, "fftw_lib") || return nothing
  file = joinpath(dirname(Base.active_project()), "LocalPreferences.toml")
  prefs = isfile(file) ? TOML.parsefile(file) : Dict{String,Any}()
  want = Dict{String,Any}("libfftw3_path" => profile["fftw_lib"])
  haskey(profile, "fftwf_lib") && (want["libfftw3f_path"] = profile["fftwf_lib"])
  get(prefs, "FFTW_jll", Dict()) == want && return nothing
  prefs["FFTW_jll"] = want
  open(io -> TOML.print(io, prefs), file, "w")
  nothing
end

# Lines for the .meta; an error when the profile wants a native build and another library is loaded.
function fftw_provenance(profile, FFTW; required)
  J = FFTW.FFTW_jll
  lines = String[]
  for (name, path, sym) in (("fftw", J.libfftw3_path, :fftw_version), ("fftwf", J.libfftw3f_path, :fftwf_version))
    h = Libdl.dlopen(path)
    version = unsafe_string(Ptr{UInt8}(Libdl.dlsym(h, sym)))
    cc = unsafe_string(Ptr{UInt8}(Libdl.dlsym(h, sym === :fftw_version ? :fftw_cc : :fftwf_cc)))
    origin = startswith(realpath(path), realpath(J.artifact_dir)) ? "FFTW_jll artifact" : "native"
    push!(lines, "$name: $origin; $version; cc: $cc; $(basename(path))")
  end
  if required
    want = get(profile, "fftw_lib", nothing)
    if want === nothing
      get(profile, "fftw", "native") == "jll" ||
        error("machine profile has no fftw_lib (native FFTW build); set it, or fftw = \"jll\" to measure the JLL")
    elseif realpath(J.libfftw3_path) != realpath(want)
      error("FFTW loads $(J.libfftw3_path), the profile wants $want")
    end
  end
  lines
end
