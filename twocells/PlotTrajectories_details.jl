# Import required packages
using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO
using Revise
using CellularDecisions

# Include helper files
include("../twocells/twocells_vis.jl")
include("../twocells/twocells_setup.jl")
include("../twocells/twocells_nonlinearsolver.jl")
include("../twocells/twocells_hittingtime.jl")
include("../utils/ctmc_vis.jl")
include("../utils/ctmc_core.jl")
include("../utils/file_utils.jl")
include("../utils/video_utils.jl")

N = 3  # Number of states - 1
λ = 20.0  # Lambda parameter
initial_state = 1  # Initial state for simulations
T = 1000.0  # Time for simulations
num_simulations = 4000  # Number of simulations

# Get state matrices, sizes, and target states
statedict,statedictinv,terminal_states,TG,TB,Tc=CellularDecisions.statematrices(N);
ni,np,ns,nt=CellularDecisions.varioussizes(N)
targetstates_good=[target_state+1 for target_state ∈ TG];  # Good target states
targetstates_bad=[target_state+1 for target_state ∈ TB];   # Bad target states
targetstates=[targetstates_good;targetstates_bad]          # All target states
startstates=[start_state+1 for start_state ∈ Tc];         # Starting states
allstates=[startstates;targetstates_good; targetstates_bad]
all_targetstates = vcat(targetstates_good, targetstates_bad)

lambda_str = replace(string(λ), "." => "_")
base_folder = joinpath(dirname(@__DIR__), "experiments", "results", "Interior_point_method_results")
folder_name = joinpath(base_folder, @sprintf("Interior_Point_Method_results_N_%d_lambda_%s", N, lambda_str))
twocell_system_filename = generate_filename(folder_name,"twocell_system")
twocell_system = CellularDecisions.load(twocell_system_filename)

N=twocell_system.internal_states
Q = twocell_system.Q_matrix
P = twocell_system.parameters

#plot heatmap of Q_opt
Q_filename = generate_filename(folder_name,"Q_matrix_heatmap")
plot_Q_with_colored_yticks(Q, N, all_targetstates, Q_filename, λ, save_plots=true)

# Simulate CTMC
println("absorbing states: ", all_targetstates)
Q_opt = copy(Q)
Q_opt_absorbing=Q_absorbing_states_maker(Q_opt, all_targetstates)
path_arr = [CellularDecisions.simulate_ctmc(Q_opt, initial_state, T, N) for i = 1:num_simulations]
path_arr_absorbing = [CellularDecisions.simulate_ctmc(Q_opt_absorbing, initial_state, T, N) for i = 1:num_simulations]
trajectories_filename = generate_filename(folder_name,"trajectories")
CellularDecisions.save(path_arr,trajectories_filename*"_not_absorbing.h5")
CellularDecisions.save(path_arr_absorbing,trajectories_filename*"_absorbing.h5")
S1_arr = CellularDecisions.load(trajectories_filename*"_not_absorbing.h5")
S2_arr = CellularDecisions.load(trajectories_filename*"_absorbing.h5")

# Plot single CTMC simulation (without absorbing states)
println("Plotting single ctmc simulation...")
ctmc_simulation_filename = generate_filename(folder_name,"single_ctmc_simulation")
p1 = plot()
to_plot = unpack(S1_arr[1])
plot_ctmc!(p1,to_plot.times, to_plot.u1, to_plot.final_time,c=:blue,linewidth=0.2)
plot_ctmc!(p1,to_plot.times, to_plot.u2, to_plot.final_time,c=:red,linewidth=0.2)
plot(p1,layout=(1,1))
savefig(p1,ctmc_simulation_filename * ".png")
#show plot
display(p1)

# Plot single CTMC simulation (with absorbing states)
println("Plotting single ctmc simulation with absorbing states...")
ctmc_simulation_filename = generate_filename(folder_name,"single_ctmc_simulation_absorbing")
p2 = plot()
to_plot = unpack(S2_arr[1])
plot_ctmc!(p2,to_plot.times, to_plot.u1, to_plot.final_time,c=:blue,linewidth=0.2)
plot_ctmc!(p2,to_plot.times, to_plot.u2, to_plot.final_time,c=:red,linewidth=0.2)
plot(p2,layout=(1,1))
savefig(p2,ctmc_simulation_filename * ".png")
#show plot
display(p2)

# Plot multiple CTMC simulations
println("Plotting multiple ctmc simulation...")
longtime_heatmap_simulation_filename = generate_filename(folder_name,"multiple_ctmc_simulation_heatmap_longtime")
plot_ctmc_multi_traj_heatmap(N, S1_arr, longtime_heatmap_simulation_filename, save_plots=true)

# Plot multiple CTMC simulations with absoribing states
println("Plotting multiple ctmc simulation with absorbing states...")
longtime_heatmap_simulation_filename = generate_filename(folder_name,"multiple_ctmc_simulation_heatmap_longtime_absorbing")
plot_ctmc_multi_traj_heatmap(N, S2_arr, longtime_heatmap_simulation_filename, save_plots=true)

# Plot invariant distribution
println("Plotting invariant ctmc heatmap...")
invariant_heatmap_simulation_filename = generate_filename(folder_name,"invariant_ctmc_heatmap")
plot_ctmc_invar_distn_heatmap(Q, N, invariant_heatmap_simulation_filename, λ, save_plots=true)


