# Helper script to initialize environment in VSCode REPL.
using Pkg
cd("src/")
Pkg.activate(".")
Pkg.instantiate()