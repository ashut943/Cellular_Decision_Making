using Random
using LinearAlgebra
using Statistics
using Plots
using Printf
using Distributions
using LightGraphs
using FileIO
using VideoIO
using LaTeXStrings
using Measures
using Revise
using CellularDecisions
using Distributed
using ExponentialUtilities
using Dates
using JSON

include("../../utils/utils.jl")
include("../../mult_cell/mult_cell.jl")
include("../../sampling_based_optimization/sampling_based_opti.jl")

#--------------------------------
#--------------------------------

N = 6
M = 7
h_error = 0.02
initial_state_array = ((1,0), (1,0), (1,0), (1,0), (1,0), (1,0), (1,0))
type_of_boundary_condition = "boundary_2"

#--------------------------------
#--------------------------------

error_str = replace(string(round(h_error*100, digits=1)), "." => "_")
base_folder = joinpath(dirname(dirname(@__DIR__)), "results", "seven_cell_results", "SGD_results_"*type_of_boundary_condition)
folder_name = joinpath(base_folder, @sprintf("trajectories_N_%d_M_%d_error_%s", N, M, error_str))
mkpath(folder_name)

plots_folder = joinpath(dirname(dirname(@__DIR__)), "plots", "seven_cell_results", type_of_boundary_condition, "N_$(N)_M_$(M)_trajectories")
mkpath(plots_folder)

println("=== Starting trajectory simulations ===")
println("N = $N, M = $M, h_error = $h_error")

#--------------------------------
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

T_horizon = 1000.0
println("AdjMat: $AdjMat")
println("T_horizon: $T_horizon")

TG_proper = [(N,0,0,0,0,0,0), (0,N,0,0,N,0,0), (0,0,N,0,0,N,0), (0,0,0,N,0,0,N), (0,N,0,N,0,N,0), (0,0,N,0,N,0,N)]

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

println("TG_proper: $TG_proper")
println("TB_proper: $(length(TB_proper)) bad states")

#--------------------------------
# Pre-computed parameter vectors
#--------------------------------

θ_vectors = Dict(
    0.02 => [0.0, 0.03295253588963976, 0.029389852058424044, 0.7030231251614495, 0.9808367795978125, 0.9795684377588251, 0.0, 0.001738214156646322, 0.0, 0.040132518419628525, 0.04818356155078206, 0.006481623816163985, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8331518414863534, 1.0, 1.0, 0.9999925017306047, 0.09335175912092401, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.16672773604385013],
    0.10 => [0.0, 0.093302274624027, 0.11843185725126298, 0.9761468934321484, 0.9952473255771057, 1.0, 0.0, 0.016745981117983002, 6.82224843288691e-6, 0.03488871856494207, 0.6578944998371575, 0.8934179979365698, 0.0, 0.023343739073633813, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9999846591733771, 1.0, 0.9999954972230595, 0.09890088258840768, 0.0, 0.0, 0.0, 2.7006761716804e-7, 1.0, 1.0, 1.0, 1.0, 7.533745440929073e-5]
)

if !haskey(θ_vectors, h_error)
    error("No pre-computed θ vector for h_error = $h_error")
end

θ_full = θ_vectors[h_error]
println("θ_full length: $(length(θ_full))")

#--------------------------------
# Run simulations (using simulate_ctmc_faster from sampling_based_funcs.jl)
#--------------------------------

P_opt_dict = Dict(i => θ_full for i in 1:M)
N_s = 20000

println("Running $N_s simulations...")

all_trajectories = []
all_times = []
good_count = 0
bad_count = 0
good_per_cell = zeros(M)
trajectory_cell_one_numbers = []
times_cell_one_numbers = []
trajectory_cell_two_numbers = []
times_cell_two_numbers = []

for sim in 1:N_s
    global good_count, bad_count
    times, states, terminal_time, is_bad = simulate_ctmc_faster(N, M, AdjMat, P_opt_dict, initial_state_array, T_horizon, TG_proper, TB_proper)
    
    trajectory = zeros(M, length(times))
    for (t_idx, state) in enumerate(states)
        for cell_idx in 1:M
            trajectory[cell_idx, t_idx] = state[cell_idx][1]
        end
    end
    
    push!(all_trajectories, trajectory)
    push!(all_times, times)
    
    if is_bad == 1
        bad_count += 1
    else
        good_count += 1
        for cell_idx in 1:M
            curr_cell_state = trajectory[cell_idx, end]
            if curr_cell_state == N
                good_per_cell[cell_idx] += 1
                if cell_idx == 1
                    push!(trajectory_cell_one_numbers, trajectory)
                    push!(times_cell_one_numbers, times)
                elseif cell_idx == 2
                    push!(trajectory_cell_two_numbers, trajectory)
                    push!(times_cell_two_numbers, times)
                end
            end
        end
    end
    
    if sim % 1000 == 0
        println("Completed $sim simulations (Good: $good_count, Bad: $bad_count)")
    end
end

println("Simulation complete! Good: $good_count, Bad: $bad_count")
println("Success rate: $(good_count/N_s)")
println("Good per cell: $good_per_cell")

#--------------------------------
# Compute mean trajectories (unconditioned)
#--------------------------------

t_grid = 0:0.5:T_horizon
mean_trajectories = zeros(M, length(t_grid))
count_trajectories = zeros(M, length(t_grid))

for sim in 1:N_s
    times = all_times[sim]
    trajectory = all_trajectories[sim]
    
    for cell_idx in 1:M
        for (t_idx, t) in enumerate(t_grid)
            state_idx = findlast(times .<= t)
            if state_idx !== nothing
                mean_trajectories[cell_idx, t_idx] += trajectory[cell_idx, state_idx]
                count_trajectories[cell_idx, t_idx] += 1
            end
        end
    end
end

for cell_idx in 1:M
    for t_idx in 1:length(t_grid)
        if count_trajectories[cell_idx, t_idx] > 0
            mean_trajectories[cell_idx, t_idx] /= count_trajectories[cell_idx, t_idx]
        end
    end
end

#--------------------------------
# Plot unconditioned trajectories
#--------------------------------

println("Creating unconditioned trajectory plot...")
colors = palette(:Set2_7)

p = plot(size=(1200, 800), dpi=150, 
         xlabel="Time", ylabel="Internal State", 
         legend=:outertop, legendfontsize=8)

for sim in 1:min(1000, N_s)
    times = all_times[sim]
    trajectory = all_trajectories[sim]
    
    for cell_idx in 1:M
        plot!(p, times, trajectory[cell_idx, :], 
              color=colors[cell_idx], alpha=0.1, 
              linewidth=0.5, label="", 
              linetype=:steppost)
    end
end

for cell_idx in 1:M
    plot!(p, t_grid, mean_trajectories[cell_idx, :], 
          color=colors[cell_idx], linewidth=3, 
          label="Cell $cell_idx (mean)", 
          linestyle=:solid)
end

plot_filename = joinpath(plots_folder, "trajectories_N$(N)_M$(M)_h_error$(h_error)_Ns$(N_s).png")
plot_filename_svg = joinpath(plots_folder, "trajectories_N$(N)_M$(M)_h_error$(h_error)_Ns$(N_s).svg")
savefig(p, plot_filename)
savefig(p, plot_filename_svg)
println("Saved: $plot_filename")

#--------------------------------
# Compute and plot conditioned trajectories
#--------------------------------

function compute_conditioned_mean(trajectory_list, times_list, M, t_grid)
    mean_traj = zeros(M, length(t_grid))
    count_traj = zeros(M, length(t_grid))
    
    for sim in 1:length(trajectory_list)
        trajectory = trajectory_list[sim]
        times = times_list[sim]
        
        for cell_idx in 1:M
            for (t_idx, t) in enumerate(t_grid)
                state_idx = findlast(times .<= t)
                if state_idx !== nothing
                    mean_traj[cell_idx, t_idx] += trajectory[cell_idx, state_idx]
                    count_traj[cell_idx, t_idx] += 1
                end
            end
        end
    end
    
    for cell_idx in 1:M
        for t_idx in 1:length(t_grid)
            if count_traj[cell_idx, t_idx] > 0
                mean_traj[cell_idx, t_idx] /= count_traj[cell_idx, t_idx]
            end
        end
    end
    
    return mean_traj
end

function plot_conditioned_trajectories(trajectory_list, times_list, mean_traj, M, t_grid, colors)
    p = plot(size=(1200, 800), dpi=150, 
             xlabel="Time", ylabel="Internal State", 
             legend=:outertop, legendfontsize=8)
    
    for sim in 1:min(1000, length(trajectory_list))
        times = times_list[sim]
        trajectory = trajectory_list[sim]
        
        for cell_idx in 1:M
            plot!(p, times, trajectory[cell_idx, :], 
                  color=colors[cell_idx], alpha=0.1, 
                  linewidth=0.5, label="", 
                  linetype=:steppost)
        end
    end
    
    for cell_idx in 1:M
        plot!(p, t_grid, mean_traj[cell_idx, :], 
              color=colors[cell_idx], linewidth=3, 
              label="Cell $cell_idx (mean)", 
              linestyle=:solid)
    end
    
    return p
end

println("Creating conditioned trajectory plots...")

mean_traj_cell1 = compute_conditioned_mean(trajectory_cell_one_numbers, times_cell_one_numbers, M, t_grid)
p1 = plot_conditioned_trajectories(trajectory_cell_one_numbers, times_cell_one_numbers, mean_traj_cell1, M, t_grid, colors)
savefig(p1, joinpath(plots_folder, "trajectories_conditioned_cell_1_N$(N)_M$(M)_h_error$(h_error).png"))
savefig(p1, joinpath(plots_folder, "trajectories_conditioned_cell_1_N$(N)_M$(M)_h_error$(h_error).svg"))
println("Saved conditioned plot for cell 1")

mean_traj_cell2 = compute_conditioned_mean(trajectory_cell_two_numbers, times_cell_two_numbers, M, t_grid)
p2 = plot_conditioned_trajectories(trajectory_cell_two_numbers, times_cell_two_numbers, mean_traj_cell2, M, t_grid, colors)
savefig(p2, joinpath(plots_folder, "trajectories_conditioned_cell_2_N$(N)_M$(M)_h_error$(h_error).png"))
savefig(p2, joinpath(plots_folder, "trajectories_conditioned_cell_2_N$(N)_M$(M)_h_error$(h_error).svg"))
println("Saved conditioned plot for cell 2")

#--------------------------------
# Parameter heatmaps
#--------------------------------

println("Creating parameter heatmaps...")

f_plus_data = zeros(N, 2)
f_plus_data[:, 1] = θ_full[1:N]
f_plus_data[:, 2] = θ_full[N+1:2*N]

f_minus_data = zeros(N, 2)
f_minus_data[:, 1] = θ_full[2*N+1:3*N]
f_minus_data[:, 2] = θ_full[3*N+1:4*N]

g_data = reshape(θ_full[4*N+1:4*N+N+1], :, 1)
k_off_data = reshape([θ_full[end]], 1, 1)

clim = (
    min(minimum(f_plus_data), minimum(f_minus_data), minimum(g_data), minimum(k_off_data)),
    max(maximum(f_plus_data), maximum(f_minus_data), maximum(g_data), maximum(k_off_data))
)

println("Parameter ranges: clim = $clim")

p_f_plus = heatmap(
    f_plus_data,
    xlabel=L"$f^+$",
    xticks=([1, 2], ["0", "1"]),
    yticks=(1:N, 0:N-1),
    ylabel="Internal State",
    color=:acton,
    size=(200, 50*N),
    titlefontsize=14,
    guidefontsize=12,
    tickfontsize=10,
    framestyle=:box,
    clims=clim,
    colorbar=true,
    title="f+ (Forward rates)"
)

p_f_minus = heatmap(
    f_minus_data,
    xlabel=L"$f^-$",
    xticks=([1, 2], ["0", "1"]),
    yticks=(1:N, 1:N),
    ylabel="Internal State",
    color=:acton,
    size=(200, 50*N),
    titlefontsize=14,
    guidefontsize=12,
    tickfontsize=10,
    framestyle=:box,
    clims=clim,
    colorbar=true,
    title="f- (Backward rates)"
)

p_rates = plot(p_f_plus, p_f_minus, layout=(1,2), size=(500, 50*N))
savefig(p_rates, joinpath(plots_folder, "parameters_rates_N$(N)_M$(M)_h_error$(h_error).svg"))

p_g = heatmap(
    g_data,
    xlabel=L"$g$",
    ylabel="Internal State",
    yticks=(1:N+1, 0:N),
    xticks=false,
    color=:acton,
    size=(100, 50*(N+1)),
    titlefontsize=14,
    guidefontsize=12,
    tickfontsize=10,
    framestyle=:box,
    clims=clim,
    colorbar=true,
    title="g (Signaling rates)"
)

for y in 1:N+1
    hline!(p_g, [y-0.3], color=:black, lw=0.3, label=nothing)
end
savefig(p_g, joinpath(plots_folder, "parameters_g_N$(N)_M$(M)_h_error$(h_error).svg"))

p_k_off = heatmap(
    k_off_data',
    xlabel=L"$k_{off}$",
    xticks=false,
    yticks=false,
    color=:acton,
    size=(100, 100),
    titlefontsize=14,
    guidefontsize=12,
    tickfontsize=10,
    framestyle=:box,
    clims=clim,
    colorbar=true,
    title="k_off (Signal decay)"
)
savefig(p_k_off, joinpath(plots_folder, "parameters_koff_N$(N)_M$(M)_h_error$(h_error).svg"))

p_combined = plot(p_f_plus, p_f_minus, p_g, p_k_off, 
                  layout=@layout([a b; c d]), 
                  size=(800, 800))
savefig(p_combined, joinpath(plots_folder, "parameters_combined_N$(N)_M$(M)_h_error$(h_error).svg"))

println("Parameter heatmaps saved!")

#--------------------------------
# Save results
#--------------------------------

results_dict = Dict(
    "N" => N,
    "M" => M,
    "h_error" => h_error,
    "T_horizon" => T_horizon,
    "N_s" => N_s,
    "good_count" => good_count,
    "bad_count" => bad_count,
    "success_rate" => good_count / N_s,
    "good_per_cell" => good_per_cell,
    "theta_full" => θ_full
)

open(joinpath(folder_name, "simulation_results.json"), "w") do io
    JSON.print(io, results_dict)
end

println("Results saved to: $folder_name")
println("Plots saved to: $plots_folder")
println("=== Simulation complete ===")
