# Import required packages
using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO

# Include helper files
include("../twocells/twocells_vis.jl")
include("../twocells/twocells_setup.jl")
include("../twocells/twocells_nonlinearsolver.jl")
include("../twocells/twocells_hittingtime.jl")
include("../utils/ctmc_vis.jl")
include("../utils/ctmc_core.jl")
include("../utils/file_utils.jl")
include("../utils/video_utils.jl")

# Set problem parameters
N = 3  # Number of states - 1
λ = 261.0  # Lambda parameter
initial_tau_val=10.0  # Initial tau value for optimization
initial_P_val=1.0  # Initial P value for optimization

# Create output directory
lambda_str = replace(string(λ), "." => "_")
base_folder = joinpath(dirname(@__DIR__), "experiments", "results", "Interior_point_method_results")
folder_name = joinpath(base_folder, @sprintf("Interior_Point_Method_results_N_%d_lambda_%s", N, lambda_str))
mkpath(folder_name)

# Get state matrices and sizes
S,Skeyer,T,TG,TB,Tc=statematrices(N);
ni,np,ns,nt=varioussizes(N)

# Define state sets
targetstates_good=[target_state+1 for target_state ∈ TG];  # Good target states
targetstates_bad=[target_state+1 for target_state ∈ TB];   # Bad target states
targetstates=[targetstates_good;targetstates_bad]          # All target states
startstates=[start_state+1 for start_state ∈ Tc];         # Starting states
allstates=[startstates;targetstates_good; targetstates_bad]
all_targetstates = vcat(targetstates_good, targetstates_bad)

# Run nonlinear solver to get optimal solution
upper_bound_tau_0,upper_bound,tau_opt,P_opt,terminationstatus=run_nonlinear_solver(N, λ, initial_P_val, initial_tau_val,false,false)

# Extract results
tau_opt_tilde = tau_opt[startstates]
lower_bound=minimum(tau_opt_tilde)

# Clean up P_opt values
P_opt_ = P_opt .* (abs.(P_opt) .>= 1e-8)  # Zero out small values
P_opt_ .= min.(P_opt_, 1.0)                # Cap at 1.0
Q_opt = Q_maker_using_M(P_opt_, N, λ, S, Skeyer)  # Generate Q matrix

#plot heatmap of Q_opt
Q_filename = generate_filename(folder_name,"Q_matrix_heatmap")
plot_Q_with_colored_yticks(Q_opt, N, all_targetstates, Q_filename, λ, save_plots=false)

# Print results for each start state
for start_state_idx in startstates
    println(S[start_state_idx-1],"->",tau_opt[start_state_idx])
end
println(terminationstatus,", Optimal: ", tau_opt[1],", Upper bound: ", upper_bound,", Lower Bound: ",lower_bound)
println("Is Q irreducible? ", is_irreducible(Q_opt))

# Plot Q matrix heatmap
Q_filename = generate_filename(folder_name,"Q_matrix_heatmap")
plot_Q_with_colored_yticks(Q_opt, N, all_targetstates, Q_filename, λ, save_plots=true)

# Simulate CTMC
initial_state = 1
T = 100.0
println("absorbing states: ", all_targetstates)
Q_opt_copy = copy(Q_opt)
Q_opt_absorbing=Q_absorbing_states_maker(Q_opt_copy, all_targetstates)
println("determinant of Q_opt_absorbing: ", det(Q_opt_absorbing))
times, states = simulate_ctmc(Q_opt, initial_state, T)
times_absorbing, states_absorbing = simulate_ctmc(Q_opt_absorbing, initial_state, T)

# Plot single CTMC simulation
println("Plotting single ctmc simulation...")
ctmc_simulation_filename = generate_filename(folder_name,"single_ctmc_simulation")
plot_ctmc_our_problem(times_absorbing, states_absorbing, T, N, ctmc_simulation_filename, λ, save_plots=true)

# Parameters for multiple simulations
num_simulations = 100
T=1000.0
initial_state = 1

# Plot multiple CTMC simulations
println("Plotting multiple ctmc simulation...")
T = 7000.0
longtime_heatmap_simulation_filename = generate_filename(folder_name,"multiple_ctmc_simulation_heatmap_longtime")
plot_ctmc_our_problem_multi(Q_opt, initial_state, T, N, num_simulations, longtime_heatmap_simulation_filename, λ, save_plots=true)

# Plot invariant distribution
println("Plotting invariant ctmc heatmap...")
invariant_heatmap_simulation_filename = generate_filename(folder_name,"invariant_ctmc_heatmap")
plot_ctmc_invar_distn_our_problem(Q_opt, N, invariant_heatmap_simulation_filename, λ, save_plots=true)

# Plot multiple CTMC simulations with absorbing states
println("Plotting multiple ctmc simulation with absorbing states...")
T = 7000.0
longtime_heatmap_simulation_filename = generate_filename(folder_name,"multiple_ctmc_simulation_heatmap_longtime_absorbing")
plot_ctmc_our_problem_multi(Q_opt_absorbing, initial_state, T, N, num_simulations, longtime_heatmap_simulation_filename, λ, save_plots=true)

# Plot invariant distribution with absorbing states
println("Plotting invariant ctmc heatmap with absorbing states...")
invariant_heatmap_simulation_filename = generate_filename(folder_name,"invariant_ctmc_heatmap_absorbing") 
plot_ctmc_invar_distn_our_problem(Q_opt_absorbing, N, invariant_heatmap_simulation_filename, λ, save_plots=true)

# Print final optimal solution details
# Print to console
println("================================================")
println("Optimal solution")
println("P_opt: ", P_opt_)
println("tau_opt_0: ", tau_opt[1])
println("------------------------------------------------")
println("F^+(.,0): ", P_opt_[1:(N)])
println("F^+(.,1): ", P_opt_[N+1:2*(N)])
println("F^-(.,0): ", P_opt_[2*(N)+1:3*(N)])
println("F^-(.,1): ", P_opt_[3*(N)+1:4*(N)])
println("G: ", P_opt_[4*(N)+1:end-1])
println("k_off: ", P_opt_[end])

# Save to txt file
optimal_values_filename = generate_filename(folder_name, "optimal_values_w_o_fixed_values.txt")
open(optimal_values_filename, "w") do io
    println(io, "tau_opt_0 = ", tau_opt[1])
    println(io, "k_off = ", P_opt_[end])
    println(io, "F+(.,0) = ", P_opt_[1:N])
    println(io, "F+(.,1) = ", P_opt_[N+1:2*N])
    println(io, "F-(.,0) = ", P_opt_[2*N+1:3*N])
    println(io, "F-(.,1) = ", P_opt_[3*N+1:4*N])
    println(io, "G = ", P_opt_[4*N+1:end-1])
end


###
# LOCALLY_SOLVED, Optimal: 4.88454023551395, Upper bound: 16.337801637343496, Lower Bound: 1.241834614843249
# Is Q irreducible? true
# absorbing states: [4, 20, 36, 52, 13, 29, 45, 61, 16, 32, 48, 64]
# Plotting single ctmc simulation...
# Plotting multiple ctmc simulation...
# Plotting invariant ctmc heatmap...
# Plotting multiple ctmc simulation with absorbing states...
# Plotting invariant ctmc heatmap with absorbing states...
# ================================================
# Optimal solution
# P_opt: [0.4320437464676692, 1.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.9999999950706491, 1.0, 1.0, 0.9999999675559776, -0.0, 1.0, 1.0, 1.0, 0.19901723495916154]
# tau_opt_0: 4.88454023551395
# ------------------------------------------------
# F^+(.,0): [0.4320437464676692, 1.0, 1.0]
# F^+(.,1): [-0.0, -0.0, -0.0]
# F^-(.,0): [-0.0, -0.0, 0.9999999950706491]
# F^-(.,1): [1.0, 1.0, 0.9999999675559776]
# G: [-0.0, 1.0, 1.0, 1.0]
# k_off: 0.19901723495916154


##
# LOCALLY_SOLVED, Optimal: 4.88454023551395, Upper bound: 16.337801637343496, Lower Bound: 1.2418346148432489
# Is Q irreducible? true
# absorbing states: [49, 51, 50, 52, 13, 15, 14, 16, 61, 63, 62, 64]
# Plotting single ctmc simulation...
# Plotting multiple ctmc simulation...
# Plotting invariant ctmc heatmap...
# Plotting multiple ctmc simulation with absorbing states...
# Plotting invariant ctmc heatmap with absorbing states...
# ================================================
# Optimal solution
# P_opt: [0.432043746467669, 1.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.9999999950706491, 1.0, 1.0, 0.9999999675559776, -0.0, 1.0, 1.0, 1.0, 0.19901723495916143]
# tau_opt_0: 4.88454023551395
# ------------------------------------------------
# F^+(.,0): [0.432043746467669, 1.0, 1.0]
# F^+(.,1): [-0.0, -0.0, -0.0]
# F^-(.,0): [-0.0, -0.0, 0.9999999950706491]
# F^-(.,1): [1.0, 1.0, 0.9999999675559776]
# G: [-0.0, 1.0, 1.0, 1.0]
# k_off: 0.19901723495916143


# LOCALLY_SOLVED, Optimal: 7.621081245815218, Upper bound: 91.44648036355383, Lower Bound: 1.60099236268152
# Is Q irreducible? true
# absorbing states: [4, 20, 36, 52, 13, 29, 45, 61, 16, 32, 48, 64]
# Plotting single ctmc simulation...
# Plotting multiple ctmc simulation...
# Plotting invariant ctmc heatmap...
# Plotting multiple ctmc simulation with absorbing states...
# Plotting invariant ctmc heatmap with absorbing states...
# ================================================
# Optimal solution
# P_opt: [0.2363716603042171, 0.7320930083303566, 0.7626819344428352, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, 1.0, 0.9999999971498481, 0.9999999978425006, -0.0, 1.0, 1.0, 1.0, 0.08912213235889899]
# tau_opt_0: 7.621081245815218
# ------------------------------------------------
# F^+(.,0): [0.2363716603042171, 0.7320930083303566, 0.7626819344428352]
# F^+(.,1): [-0.0, -0.0, -0.0]
# F^-(.,0): [-0.0, -0.0, 1.0]
# F^-(.,1): [1.0, 0.9999999971498481, 0.9999999978425006]
# G: [-0.0, 1.0, 1.0, 1.0]
# k_off: 0.08912213235889899


# LOCALLY_SOLVED, Optimal: 7.6210812458152155, Upper bound: 91.4464803635558, Lower Bound: 1.6009923626814664
# Is Q irreducible? true
# absorbing states: [49, 51, 50, 52, 13, 15, 14, 16, 61, 63, 62, 64]
# Plotting single ctmc simulation...
# Plotting multiple ctmc simulation...
# Plotting invariant ctmc heatmap...
# Plotting multiple ctmc simulation with absorbing states...
# Plotting invariant ctmc heatmap with absorbing states...
# ================================================
# Optimal solution
# P_opt: [0.23637166030421594, 0.7320930083303432, 0.7626819344428606, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, 1.0, 0.9999999971498499, 0.9999999978425046, -0.0, 1.0, 1.0, 1.0, 0.08912213235889839]
# tau_opt_0: 7.6210812458152155
# ------------------------------------------------
# F^+(.,0): [0.23637166030421594, 0.7320930083303432, 0.7626819344428606]
# F^+(.,1): [-0.0, -0.0, -0.0]
# F^-(.,0): [-0.0, -0.0, 1.0]
# F^-(.,1): [1.0, 0.9999999971498499, 0.9999999978425046]
# G: [-0.0, 1.0, 1.0, 1.0]
# k_off: 0.08912213235889839
