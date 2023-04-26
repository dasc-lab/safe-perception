module SafePerception
# Make sure we use the system-installed Python backend
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/usr/bin/python3"

greet() = print("Loading SafePerception")

include("ingest.jl")
include("matching.jl")
include("mapping.jl")
include("PoseEstimation.jl")
include("utils.jl")

using .PoseEstimation
export generate_synthetic_data
end # module SafePerception
