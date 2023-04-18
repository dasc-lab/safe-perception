@time include("test_mapping_multiple.jl")
run_test()
@time run_test()
# @profview plot_all()