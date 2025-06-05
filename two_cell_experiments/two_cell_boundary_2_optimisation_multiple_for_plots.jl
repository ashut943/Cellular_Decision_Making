# Import required packages
using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO
using Revise
using CellularDecisions_final

# Include helper files
include("../mult_cell/mult_cell_setup.jl")
include("../mult_cell/mult_cell_nonlinear.jl")
include("../mult_cell/mult_cell_hittingtime.jl")
include("../mult_cell/mult_cell_hittingprob.jl")
include("../utils/file_utils.jl")
include("../utils/ctmc_vis.jl")
#--------------------------------
N = 3 # Number of states - 1s
M_cell = 2
λ = 0.0# Lambda parameter
initial_tau_val=0.0# Initial tau value fo0r optimization
initial_P_val=1.0# Initial P value for optimization
initial_state_array = ((1,0),(1,0))  # Initial state for simulations
type_of_boundary_condition="boundary_2"#for saving the results
# 1.5009999990192666
# 1.5003554770866048
λ_list=[Float64(i) for i in 0.0:1.0:200.0]

# initial_state_array = ((0,0),(0,0))  # Initial state for simulations
# type_of_boundary_condition="boundary_1"#for saving the results
#--------------------------------
# Create output directory
lambda_str = replace(string(λ), "." => "_")
base_folder = joinpath(dirname(@__DIR__), "experiments", "two_cell_results", "Interior_point_method_results_"*type_of_boundary_condition)
folder_name = joinpath(base_folder, @sprintf("Interior_Point_Method_results_N_%d_lambda_%s", N, lambda_str))
mkpath(folder_name)

folder_name_for_plots="./plots/two_cell_results/"*type_of_boundary_condition*"/"*"N_$(N)_lambda_$(lambda_str)"*"_"
mkpath(folder_name_for_plots)
#--------------------------------

# Get state matrices, sizes, and target states
statedict,statedictinv,terminal_states,TG,TB,Tc=CellularDecisions_final.statematrices(N, M_cell, type_of_boundary_condition);
ni,np=CellularDecisions_final.varioussizes(N, M_cell);
ns=length(Tc);
targetstates_good=[target_state+1 for target_state ∈ TG];  # Good target states
targetstates_bad=[target_state+1 for target_state ∈ TB];   # Bad target states
targetstates=[targetstates_good;targetstates_bad];          # All target states
startstates=[start_state+1 for start_state ∈ Tc];         # Starting states (basically any state that is not one of the target states)
allstates=[startstates;targetstates_good; targetstates_bad];
all_targetstates = vcat(targetstates_good, targetstates_bad);

# initial_f_plus_array_0=[0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0];
# initial_f_plus_array_1=[0.0,0.0,0.0,0.1,0.0,1.0,.0,0.0];
# initial_f_minus_array_0=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# initial_f_minus_array_1=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
# initial_g_array=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# initial_k_off_array=[0.0]


# initial_f_plus_array_0=[initial_P_val for i in 1:N];
# initial_f_minus_array_0=[initial_P_val for i in 1:N];
# initial_f_plus_array_1=[initial_P_val for i in 1:N];
# initial_f_minus_array_1=[initial_P_val for i in 1:N];
# initial_g_array=[initial_P_val for i in 1:N+1];
# initial_k_off_array=[initial_P_val]


initial_f_plus_array_0=rand(N); 
initial_f_minus_array_0=rand(N);
initial_f_plus_array_1=rand(N);
initial_f_minus_array_1=rand(N);
initial_g_array=rand(N+1);
initial_k_off_array=[rand()]

# initial_f_plus_array_0=[-0.0, 0.6646405349455765, 1.0, 1.0];
# initial_f_plus_array_1=[-0.0, 1.8532715046941113e-8, 0.5648457791833867, 0.9999999221559852];
# initial_f_minus_array_0=[1.0, 1.2226410107339407e-7, -0.0, -0.0];
# initial_f_minus_array_1=[0.9999999707941578, 0.9999998830386608, 2.008390609482969e-6, -0.0];
# initial_g_array=[0.0, 1.0, 1.0, 1.0,1.0]
# initial_k_off_array=[0.0]

# initial_f_plus_array_0=[0.0 for i in 1:N];
# initial_f_minus_array_0=[0.0 for i in 1:N];
# initial_f_plus_array_1=[0.0 for i in 1:N];
# initial_f_minus_array_1=[0.0 for i in 1:N];
# initial_g_array=[0.0 for i in 1:N+1];
# initial_k_off_array=[0.0]

initial_P_val_array = vcat(initial_f_plus_array_0, initial_f_plus_array_1, initial_f_minus_array_0, initial_f_minus_array_1, initial_g_array, initial_k_off_array)
initial_tau_val_array=[initial_tau_val for i in 1:ns]
initial_state=statedictinv[initial_state_array]+1

#find the tau_opt for the initial_state
Q_initial = Q_maker(Dict(1=>initial_P_val_array, 2=>initial_P_val_array), N, M_cell, statedict, statedictinv)  # Generate Q matrix
Q_initial_absorbing=Q_absorbing_states_maker(Q_initial, all_targetstates)
initial_tau_opt=hitting_time_mod(Q_initial_absorbing,targetstates_good,targetstates_bad,startstates,λ)
println("initial_tau_opt: ", initial_tau_opt[initial_state])
println("--------------------------------")
# Run nonlinear solver to get optimal solution
upper_bound_tau_0,upper_bound,tau_opt,P_opt_dict_global_temp,terminationstatus=run_nonlinear_solver(N, M_cell, λ, initial_state, Dict(1=>initial_P_val_array, 2=>initial_P_val_array), initial_tau_val_array,Dict(1=>false,2=>false),false, type_of_boundary_condition)
println("--------------------------------")
println("tau_opt: ", tau_opt[initial_state])
println("terminationstatus: ", terminationstatus)
# println("P1_opt: ", P_opt_dict_global_temp[1])
# println("P2_opt: ", P_opt_dict_global_temp[2])


if terminationstatus == MOI.LOCALLY_INFEASIBLE || terminationstatus == MOI.INFEASIBLE || terminationstatus == MOI.NUMERICAL_ERROR || terminationstatus == MOI.OTHER_ERROR
    error("Local optimization failed with status: $terminationstatus")
end

# Extract results
tau_opt_tilde = tau_opt[startstates]
lower_bound=minimum(tau_opt_tilde)

# Clean up P_opt values
P1_opt_ = P_opt_dict_global_temp[1] .* (abs.(P_opt_dict_global_temp[1]) .>= 1e-8)  # Zero out small values
P1_opt_ .= min.(P1_opt_, 1.0)                # Cap at 1.0
P2_opt_ = P_opt_dict_global_temp[2] .* (abs.(P_opt_dict_global_temp[2]) .>= 1e-8)  # Zero out small values
P2_opt_ .= min.(P2_opt_, 1.0)                # Cap at 1.0
Q_opt = Q_maker(Dict(1=>P1_opt_, 2=>P2_opt_), N, M_cell, statedict, statedictinv)  # Generate Q matrix
Q_opt_absorbing=Q_absorbing_states_maker(Q_opt, all_targetstates)
#save the twocell_system 
parameters_opt_1 = CellularDecisions_final.parameter_vector_to_parameters(P1_opt_, N)
parameters_opt_2 = CellularDecisions_final.parameter_vector_to_parameters(P2_opt_, N)

parameters_opt_dict=Dict(1=>parameters_opt_1,2=>parameters_opt_2);

#--------------------------------
#saving the optimal values
println("================================================")
hitting_mod_values=hitting_time_mod(Q_opt_absorbing,targetstates_good,targetstates_bad,startstates,0.0)
hitting_mod_values_modified=hitting_time_mod(Q_opt_absorbing,targetstates_good,targetstates_bad,startstates,λ)
h_global=hitting_prob_mod(Q_opt_absorbing,targetstates_good,targetstates_bad,startstates,λ)
initial_state_index=findfirst(x -> x == initial_state, startstates) 

#printing the results
println("================================================")
println("Optimal solution")
println("F_1^+: ", parameters_opt_1.fp)
println("F_1^-: ", parameters_opt_1.fn)
println("F_2^+: ", parameters_opt_2.fp)
println("F_2^-: ", parameters_opt_2.fn)
println("G_1: ", parameters_opt_1.gp)
println("G_2: ", parameters_opt_2.gp)
println("k_off_1: ", parameters_opt_1.koff)
println("k_off_2: ", parameters_opt_2.koff)

# Save to txt file
# optimal_values_filename = generate_filename(folder_name, "optimal_values_w_o_fixed_values.txt")
# open(optimal_values_filename, "w") do io
#     println(io, "tau_opt_0 = ", tau_opt[initial_state])
#     println(io, "k_off_1 = ", P1_opt_[end])
#     println(io, "F_1^+(.,0) = ", P1_opt_[1:N])
#     println(io, "F_1^+(.,1) = ", P1_opt_[N+1:2*N])
#     println(io, "F_1^-(.,0) = ", P1_opt_[2*N+1:3*N])
#     println(io, "F_1^-(.,1) = ", P1_opt_[3*N+1:4*N])
#     println(io, "G_1 = ", P1_opt_[4*N+1:end-1])
#     println(io, "k_off_2 = ", P2_opt_[end])
#     println(io, "F_2^+(.,0) = ", P2_opt_[1:N])
#     println(io, "F_2^+(.,1) = ", P2_opt_[N+1:2*N])
#     println(io, "F_2^-(.,0) = ", P2_opt_[2*N+1:3*N])
#     println(io, "F_2^-(.,1) = ", P2_opt_[3*N+1:4*N])
#     println(io, "G_2 = ", P2_opt_[4*N+1:end-1])
# end
println("================================================")
println("λ: ", λ)
println("GLOBAL:")
println("tau_opt_global: ", tau_opt[initial_state])
println("h_global: ", h_global[initial_state_index])
println("Time, global: ",tau_opt[initial_state]-λ*h_global[initial_state_index])
println("hitting_time: ", hitting_mod_values[initial_state_index])
println("hitting_time_modified: ", hitting_mod_values_modified[initial_state_index])
println("================================================")