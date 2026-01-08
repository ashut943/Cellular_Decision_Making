using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO, LaTeXStrings
using Measures
using Revise
using CellularDecisions
using Distributed
using JLD2
using Statistics

include("../../../mult_cell/mult_cell.jl")
include("../../../utils/utils.jl")
include("../../../sampling_based_optimization/sampling_based_opti.jl")

#--------------------------------
# Configuration
#--------------------------------

N = 6
M = 3
initial_state_array = ((1,0),(1,0),(1,0))
type_of_boundary_condition = "boundary_2"

#--------------------------------
# Output directories
#--------------------------------
plots_folder = joinpath(dirname(dirname(dirname(@__DIR__))), "plots", "three_cell_results", type_of_boundary_condition, "N_$(N)_pareto_front")
mkpath(plots_folder)

#--------------------------------
# Load sampled data
#--------------------------------

error_folder = joinpath(dirname(dirname(dirname(@__DIR__))), "results", "three_cell_results", "combined_sampled_points_"*type_of_boundary_condition)
@load joinpath(error_folder, "error_rate_all.jld2") overall_error_rate_array
@load joinpath(error_folder, "speed_all.jld2") overall_speed_array
errors_batch1 = overall_error_rate_array
speeds_batch1 = overall_speed_array

@load joinpath(error_folder, "overall_error_rate_array_100_500.jld2") overall_error_rate_array
@load joinpath(error_folder, "overall_speed_array_100_500.jld2") overall_speed_array
errors_batch2 = overall_error_rate_array
speeds_batch2 = overall_speed_array

@load joinpath(error_folder, "overall_error_rate_array_500_2_again.jld2") overall_error_rate_array
@load joinpath(error_folder, "overall_speed_array_500_2_again.jld2") overall_speed_array
errors_batch3 = overall_error_rate_array
speeds_batch3 = overall_speed_array

errors_combined = [errors_batch1; errors_batch2; errors_batch3]
speeds_combined = [speeds_batch1; speeds_batch2; speeds_batch3]

# Filter samples
sampled_errors = Float64[]
sampled_speeds = Float64[]
for i in 1:length(errors_combined)
    if speeds_combined[i] > 11/6 && speeds_combined[i] <= 15.0 && errors_combined[i] < 0.99
        push!(sampled_errors, errors_combined[i])
        push!(sampled_speeds, speeds_combined[i])
    end
end

println("Filtered samples: $(length(sampled_errors))")
println("Min error rate: $(minimum(sampled_errors)) at speed: $(sampled_speeds[argmin(sampled_errors)])")
println("Max error rate: $(maximum(sampled_errors)) at speed: $(sampled_speeds[argmax(sampled_errors)])")

#--------------------------------
# Load Pareto front data from optimization results
#--------------------------------
h_error_list = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]

pareto_errors = Float64[]
pareto_speeds = Float64[]

for h_error in h_error_list
    println("Loading h_error = $h_error")
    error_str = replace(string(round(h_error*100, digits=1)), "." => "_")
    base_folder = joinpath(dirname(dirname(dirname(@__DIR__))), "results", "three_cell_results", "Interior_point_method_results_"*type_of_boundary_condition)
    folder_name = joinpath(base_folder, @sprintf("Interior_Point_Method_results_N_%d_error_fix_%s", N, error_str))
    
    threecell_system_filename = generate_filename(folder_name, "threecell_system_global")
    threecell_system_global = CellularDecisions.load(threecell_system_filename)

    statedict = threecell_system_global.state_dict
    statedictinv = threecell_system_global.state_dict_inv
    TG = threecell_system_global.T_good
    TB = threecell_system_global.T_bad
    Tc = threecell_system_global.Tc
    
    targetstates_good = [t+1 for t in TG]
    targetstates_bad = [t+1 for t in TB]
    startstates = [s+1 for s in Tc]
    initial_state = statedictinv[initial_state_array] + 1

    Q_opt = threecell_system_global.Q_matrix
    hitting_times = hitting_time_mod(Q_opt, targetstates_good, targetstates_bad, startstates, 0.0)
    hitting_probs = hitting_prob_mod(Q_opt, targetstates_good, targetstates_bad, startstates, 0.0)
    
    println("  error = $(hitting_probs[initial_state]), speed = $(hitting_times[initial_state])")
    push!(pareto_errors, hitting_probs[initial_state])
    push!(pareto_speeds, hitting_times[initial_state])
end

#--------------------------------
# Simulation setup for SGD strategies
#--------------------------------

AdjMat = zeros(Int, M, M)
AdjMat[1,2] = 1; AdjMat[1,3] = 1
AdjMat[2,1] = 1; AdjMat[2,3] = 1
AdjMat[3,1] = 1; AdjMat[3,2] = 1

T_horizon = 100.0

TG_proper = [(N,0,0), (0,N,0), (0,0,N)]
TB_proper = []
for u_1 in [0,N], u_2 in [0,N], u_3 in [0,N]
    curr = (u_1, u_2, u_3)
    if curr ∉ TG_proper && curr ∉ TB_proper
        push!(TB_proper, curr)
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

strategies_combined_folder = joinpath(dirname(dirname(dirname(@__DIR__))), "results", "three_cell_results", "SGD_results_combined_"*type_of_boundary_condition)
strategies_dict = parse_strategies_file(joinpath(strategies_combined_folder, "all_strategies_approx_M3.txt"))
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

savefig(p, joinpath(plots_folder, "pareto_front_with_samples.png"))
savefig(p, joinpath(plots_folder, "pareto_front_with_samples.svg"))

println("Plots saved to: $plots_folder")
