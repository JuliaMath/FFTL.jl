# Machine profiles. A profile is harness/machines/<id>.toml (public: templates without host names)
# or, which takes precedence, bench/machines.local/<id>.toml (private: hosts, paths and load limits
# of your own machines, not versioned).

const MACHINES_DIR = joinpath(@__DIR__, "machines")
const LOCAL_MACHINES_DIR = joinpath(@__DIR__, "..", "machines.local")

profile_path(id::AbstractString) = begin
  local_path = joinpath(LOCAL_MACHINES_DIR, id * ".toml")
  isfile(local_path) ? local_path : joinpath(MACHINES_DIR, id * ".toml")
end
