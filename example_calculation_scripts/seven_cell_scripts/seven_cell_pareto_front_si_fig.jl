using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO, LaTeXStrings
using Measures
using Revise
using CellularDecisions
using Distributed
using JLD2
using Statistics

include("../../mult_cell/mult_cell.jl")
include("../../utils/utils.jl")
include("../../sampling_based_optimization/sampling_based_opti.jl")

#--------------------------------
# Configuration
#--------------------------------

N = 6
M = 7
initial_state_array = ((1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0))
type_of_boundary_condition = "boundary_2"

#--------------------------------
# Output directories
#--------------------------------
plots_folder = joinpath(dirname(dirname(dirname(@__DIR__))), "plots", "seven_cell_results", type_of_boundary_condition, "N_$(N)_pareto_front")
mkpath(plots_folder)

#--------------------------------
# Simulation setup for SGD strategies
#--------------------------------


AdjMat = zeros(Int, M, M)

AdjMat[1,2] = 1
AdjMat[1,3] = 1
AdjMat[1,4] = 1
AdjMat[1,5] = 1
AdjMat[1,6] = 1
AdjMat[1,7] = 1

AdjMat[2,1] = 1
AdjMat[2,3] = 1
AdjMat[2,7] = 1

AdjMat[3,1] = 1
AdjMat[3,2] = 1
AdjMat[3,4] = 1

AdjMat[4,1] = 1
AdjMat[4,3] = 1
AdjMat[4,5] = 1

AdjMat[5,1] = 1
AdjMat[5,4] = 1
AdjMat[5,6] = 1

AdjMat[6,1] = 1
AdjMat[6,5] = 1
AdjMat[6,7] = 1

AdjMat[7,1] = 1
AdjMat[7,2] = 1
AdjMat[7,6] = 1

T_horizon = 300.0

TG_proper = [(N,0,0,0,0,0,0),(0,N,0,0,N,0,0), (0,0,N,0,0,N,0), (0,0,0,N,0,0,N), (0,N,0,N,0,N,0), (0,0,N,0,N,0,N)]

TB_proper = []
for u_1 in [0, N]
    for u_2 in [0, N]
        for u_3 in [0, N]
            for u_4 in [0, N]
                for u_5 in [0, N]
                    for u_6 in [0, N]
                        for u_7 in [0, N]
                            curr = (u_1, u_2, u_3, u_4, u_5, u_6, u_7)
                            if curr ∉ TG_proper && curr ∉ TB_proper
                                push!(TB_proper, curr)
                            end
                        end
                    end
                end
            end
        end
    end
end

#--------------------------------
# Parse strategies file
#--------------------------------
function parse_strategies_file(filename)
    strategies = Dict{Float64, Vector{Float64}}()
    lines = readlines(filename)
    i = 1
    
    while i <= length(lines)
        line = strip(lines[i])
        if startswith(line, "Strategy #") && contains(line, "h_error = ")
            h_error_match = match(r"h_error = ([0-9.]+):", line)
            if h_error_match !== nothing
                h_error_val = parse(Float64, h_error_match.captures[1])
                while i <= length(lines)
                    i += 1
                    if contains(lines[i], "θ_full (used for all M cells in simulation):")
                        i += 1
                        while i <= length(lines) && strip(lines[i]) != "["
                            i += 1
                        end
                        i += 1
                        theta_values = Float64[]
                        while i <= length(lines)
                            curr_line = strip(lines[i])
                            if curr_line == "]"
                                break
                            end
                            val_str = replace(curr_line, "," => "")
                            if !isempty(val_str)
                                push!(theta_values, parse(Float64, val_str))
                            end
                            i += 1
                        end
                        strategies[h_error_val] = theta_values
                        println("Loaded strategy for h_error = $h_error_val ($(length(theta_values)) params)")
                        break
                    end
                end
            end
        end
        i += 1
    end
    return strategies
end

strategies_combined_folder = joinpath(dirname(dirname(dirname(@__DIR__))), "results", "seven_cell_results", "SGD_results_combined_"*type_of_boundary_condition)
strategies_dict = parse_strategies_file(joinpath(strategies_combined_folder, "all_strategies_approx_M7.txt"))
println("Loaded $(length(strategies_dict)) strategies")

#--------------------------------
# Run simulations for each strategy
#--------------------------------
sgd_mean_times = Float64[]
sgd_errors = Float64[]
sgd_std_times = Float64[]
sgd_std_errors = Float64[]
n_sims = 10000

for h_error in keys(strategies_dict)
    θ_full = strategies_dict[h_error]
    P_opt_dict = Dict(i => θ_full for i in 1:M)
    
    println("Running $n_sims simulations for h_error = $h_error...")
    
    bad_count = 0
    times = Float64[]
    bad_arr = zeros(Int, n_sims)
    
    for sim in 1:n_sims
        _, _, terminal_time, is_bad = simulate_ctmc_faster(N, M, AdjMat, P_opt_dict, initial_state_array, T_horizon, TG_proper, TB_proper)
        push!(times, terminal_time)
        bad_arr[sim] = is_bad
        if is_bad == 1
            bad_count += 1
        end
        if sim % 1000 == 0
            println("  Completed $sim simulations (Bad: $bad_count)")
        end
    end
    
    push!(sgd_mean_times, mean(times))
    push!(sgd_errors, bad_count / n_sims)
    push!(sgd_std_times, std(times) / sqrt(n_sims))
    push!(sgd_std_errors, std(bad_arr) / sqrt(n_sims))
    println("  Error rate: $(bad_count/n_sims), mean time: $(mean(times))")
end

#--------------------------------
# Plotting
#--------------------------------
sorted_idx = sortperm(sgd_errors)
sorted_errors = sgd_errors[sorted_idx]
sorted_means = sgd_mean_times[sorted_idx]
sorted_std_times = sgd_std_times[sorted_idx]
sorted_std_errors = sgd_std_errors[sorted_idx]
yerr = 1.96 .* sorted_std_times
xerr = 1.96 .* sorted_std_errors

p = scatter(sampled_errors, sampled_speeds, 
    xlabel="Error Rate", 
    ylabel="Mean First Passage Time", 
    title="Trade off between accuracy and speed", 
    legend=false, 
    markersize=3,
    size=(800,600),
    alpha=0.25,
    color="#8BA6C1"
)

vline!([0.02], color="#4a7494", linestyle=:dash, label=false, linewidth=2)
vline!([0.1], color="#c4806b", linestyle=:dash, label=false, linewidth=2)

plot!(pareto_errors, pareto_speeds, legend=false, color="#3a2c52", markersize=3)
scatter!(pareto_errors, pareto_speeds, color="#3a2c52", legend=false, markersize=3)

plot!(sorted_errors, sorted_means, linestyle=:dash, lw=2, color="#006666")
scatter!(sorted_errors, sorted_means, ms=2, xerror=xerr, yerror=yerr, color="#006666", label="")

savefig(p, joinpath(plots_folder, "pareto_front.png"))
savefig(p, joinpath(plots_folder, "pareto_front.svg"))

println("Plots saved to: $plots_folder")
