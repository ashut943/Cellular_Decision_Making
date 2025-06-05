# Import required packages
using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO
using Revise
using CellularDecisions_final

# Include helper files
include("../mult_cell/mult_cell_setup.jl")
include("../mult_cell/mult_cell_nonlinear.jl")
include("../mult_cell/mult_cell_hittingtime.jl")
include("../utils/file_utils.jl")
include("../utils/ctmc_vis.jl")
#--------------------------------
N = 3  # Number of states - 1
M_cell = 3
λ = 30.0  # Lambda parameter
initial_tau_val=10.0# Initial tau value for optimization
initial_P_val=1.0 # Initial P value for optimization
initial_state_array = ((1,0),(1,0),(1,0))  # Initial state for simulations
type_of_boundary_condition="boundary_2"#for saving the results

# initial_tau_val=8.0
# initial_state_array = ((0,0),(0,0),(0,0))  # Initial state for simulations
# type_of_boundary_condition="boundary_1"#for saving the results
#--------------------------------
# Create output directory
lambda_str = replace(string(λ), "." => "_")
base_folder = joinpath(dirname(@__DIR__), "experiments", "three_cell_results", "Interior_point_method_results_"*type_of_boundary_condition)
folder_name = joinpath(base_folder, @sprintf("Interior_Point_Method_results_N_%d_lambda_%s", N, lambda_str))
mkpath(folder_name)

folder_name_for_plots="./plots/three_cell_results/"*type_of_boundary_condition*"/"*"N_$(N)_lambda_$(lambda_str)"*"_"
mkpath(folder_name_for_plots)
#--------------------------------

# Get state matrices, sizes, and target states
statedict,statedictinv,terminal_states,TG,TB,Tc=CellularDecisions_final.statematrices(N, M_cell, type_of_boundary_condition);
ni,np=CellularDecisions_final.varioussizes(N, M_cell);
ns=length(Tc);
targetstates_good=[target_state+1 for target_state ∈ TG];  # Good target states
targetstates_bad=[target_state+1 for target_state ∈ TB];   # Bad target states
targetstates=[targetstates_good;targetstates_bad];          # All target states
startstates=[start_state+1 for start_state ∈ Tc];         # Starting states
allstates=[startstates;targetstates_good; targetstates_bad];
all_targetstates = vcat(targetstates_good, targetstates_bad);


initial_f_plus_array_0=[initial_P_val for i in 1:N];
initial_f_minus_array_0=[initial_P_val for i in 1:N];
initial_f_plus_array_1=[initial_P_val for i in 1:N];
initial_f_minus_array_1=[initial_P_val for i in 1:N];
initial_g_array=[initial_P_val for i in 1:N+1];
initial_k_off_array=[initial_P_val]

initial_P_val_array = vcat(initial_f_plus_array_0, initial_f_plus_array_1, initial_f_minus_array_0, initial_f_minus_array_1, initial_g_array, initial_k_off_array)
initial_tau_val_array=[initial_tau_val for i in 1:ns]
initial_state=statedictinv[initial_state_array]+1

#find the tau_opt for the initial_state
Q_initial = Q_maker(Dict(1=>initial_P_val_array, 2=>initial_P_val_array, 3=>initial_P_val_array), N, M_cell, statedict, statedictinv)  # Generate Q matrix
Q_initial_absorbing=Q_absorbing_states_maker(Q_initial, all_targetstates)
initial_tau_opt=hitting_time_mod(Q_initial_absorbing,targetstates_good,targetstates_bad,startstates,λ)
println("initial_tau_opt: ", initial_tau_opt[initial_state])
println("--------------------------------")
# Run nonlinear solver to get optimal solution
upper_bound_tau_0,upper_bound,tau_opt,P_opt_dict_global_temp,terminationstatus=run_nonlinear_solver(N, M_cell, λ, initial_state, Dict(1=>initial_P_val_array, 2=>initial_P_val_array, 3=>initial_P_val_array), initial_tau_val_array,Dict(1=>false,2=>false,3=>false),false, type_of_boundary_condition)
println("tau_opt: ", tau_opt[initial_state])
println("P1_opt: ", P_opt_dict_global_temp[1])
println("P2_opt: ", P_opt_dict_global_temp[2])
println("P3_opt: ", P_opt_dict_global_temp[3])
println("terminationstatus: ", terminationstatus)

# Extract results
tau_opt_tilde = tau_opt[startstates]
lower_bound=minimum(tau_opt_tilde)

# Clean up P_opt values
P1_opt_ = P_opt_dict_global_temp[1] .* (abs.(P_opt_dict_global_temp[1]) .>= 1e-7)  # Zero out small values
P1_opt_ .= min.(P1_opt_, 1.0)                # Cap at 1.0
P2_opt_ = P_opt_dict_global_temp[2] .* (abs.(P_opt_dict_global_temp[2]) .>= 1e-7)  # Zero out small values
P2_opt_ .= min.(P2_opt_, 1.0)                # Cap at 1.0
P3_opt_ = P_opt_dict_global_temp[3] .* (abs.(P_opt_dict_global_temp[3]) .>= 1e-7)  # Zero out small values
P3_opt_ .= min.(P3_opt_, 1.0)                # Cap at 1.0
Q_opt = Q_maker(Dict(1=>P1_opt_, 2=>P2_opt_, 3=>P3_opt_), N, M_cell, statedict, statedictinv)  # Generate Q matrix
Q_opt_absorbing=Q_absorbing_states_maker(Q_opt, all_targetstates)
#save the twocell_system 
parameters_opt_1 = CellularDecisions_final.parameter_vector_to_parameters(P1_opt_, N)
parameters_opt_2 = CellularDecisions_final.parameter_vector_to_parameters(P2_opt_, N)
parameters_opt_3 = CellularDecisions_final.parameter_vector_to_parameters(P3_opt_, N)
parameters_opt_dict=Dict(1=>parameters_opt_1,2=>parameters_opt_2,3=>parameters_opt_3);

twocell_system = CellularDecisions_final.build_cell_system(N, M_cell, Q_opt_absorbing, parameters_opt_dict, type_of_boundary_condition)
# println(twocell_system.state_dict_inv)
twocell_system_filename = generate_filename(folder_name,"threecell_system_global")
CellularDecisions_final.save(twocell_system,twocell_system_filename)

#--------------------------------
#saving the optimal values
println("================================================")
hitting_mod_values=hitting_time_mod(Q_opt_absorbing,targetstates_good,targetstates_bad,startstates,λ)
#find the index of initial_state in startstates
initial_state_index=findfirst(x -> x == initial_state, startstates) 
println("hitting_time: ", hitting_mod_values[initial_state_index])
println("lambda: ", λ)
# Print to console
println("================================================")
println("Optimal solution")
println("P1_opt: ", P_opt_dict_global_temp[1])
println("P2_opt: ", P_opt_dict_global_temp[2])
println("P3_opt: ", P_opt_dict_global_temp[3])
println("tau_opt_0: ", tau_opt[initial_state])
println("------------------------------------------------")
println("F_1^+: ", parameters_opt_1.fp)
println("F_1^-: ", parameters_opt_1.fn)
println("F_2^+: ", parameters_opt_2.fp)
println("F_2^-: ", parameters_opt_2.fn)
println("F_3^+: ", parameters_opt_3.fp)
println("F_3^-: ", parameters_opt_3.fn)
println("G_1: ", parameters_opt_1.gp)
println("G_2: ", parameters_opt_2.gp)
println("G_3: ", parameters_opt_3.gp)
println("k_off_1: ", parameters_opt_1.koff)
println("k_off_2: ", parameters_opt_2.koff)
println("k_off_3: ", parameters_opt_3.koff)

# Save to txt file
optimal_values_filename = generate_filename(folder_name, "optimal_values_w_o_fixed_values.txt")
open(optimal_values_filename, "w") do io
    println(io, "tau_opt_0 = ", tau_opt[initial_state])
    println(io, "k_off_1 = ", P1_opt_[end])
    println(io, "F_1^+(.,0) = ", P1_opt_[1:N])
    println(io, "F_1^+(.,1) = ", P1_opt_[N+1:2*N])
    println(io, "F_1^-(.,0) = ", P1_opt_[2*N+1:3*N])
    println(io, "F_1^-(.,1) = ", P1_opt_[3*N+1:4*N])
    println(io, "G_1 = ", P1_opt_[4*N+1:end-1])
    println(io, "k_off_2 = ", P2_opt_[end])
    println(io, "F_2^+(.,0) = ", P2_opt_[1:N])
    println(io, "F_2^+(.,1) = ", P2_opt_[N+1:2*N])
    println(io, "F_2^-(.,0) = ", P2_opt_[2*N+1:3*N])
    println(io, "F_2^-(.,1) = ", P2_opt_[3*N+1:4*N])
    println(io, "G_2 = ", P2_opt_[4*N+1:end-1])
    println(io, "k_off_3 = ", P3_opt_[end])
    println(io, "F_3^+(.,0) = ", P3_opt_[1:N])
    println(io, "F_3^+(.,1) = ", P3_opt_[N+1:2*N])
    println(io, "F_3^-(.,0) = ", P3_opt_[2*N+1:3*N])
    println(io, "F_3^-(.,1) = ", P3_opt_[3*N+1:4*N])
    println(io, "G_3 = ", P3_opt_[4*N+1:end-1])
end

#--------------------------------
#plotting a few trajectories
T=25.0
num_simulations=1000

# Q_opt_absorbing=Q_absorbing_states_maker(Q, all_targetstates)
println("Simulating CTMC")
S_arr  = [CellularDecisions_final.simulate_ctmc(Q_opt_absorbing, initial_state, T,N, M_cell, type_of_boundary_condition) for i = 1:num_simulations]
println("Done simulating CTMC")
targetstates_good_1 = targetstates_good[[statedict[t-1][2][1] ==0 && statedict[t-1][3][1] ==0  for t in targetstates_good]]
targetstates_good_2 = targetstates_good[[statedict[t-1][1][1] ==0 && statedict[t-1][3][1] ==0  for t in targetstates_good]]
targetstates_good_3 = targetstates_good[[statedict[t-1][1][1] ==0 && statedict[t-1][2][1] ==0  for t in targetstates_good]]

terminal_classes = [CellularDecisions_final.terminal_class(path, [targetstates_good_1,targetstates_good_2,targetstates_good_3], targetstates_bad) for path in S_arr]

success_trajectories = unpack.(S_arr[terminal_classes .!= -1])
failed_trajectories = unpack.(S_arr[terminal_classes .== -1])

failed_trajectories = (S_arr[terminal_classes .== -1])
success_trajectories = (S_arr[terminal_classes .== 1 .|| terminal_classes .== 2 .|| terminal_classes .== 3])
success_trajectories_1 = (S_arr[terminal_classes .== 1])
success_trajectories_2 = (S_arr[terminal_classes .== 2])
success_trajectories_3 = (S_arr[terminal_classes .== 3])

println("Success trajectories 1: ", length(success_trajectories_1))
println("Success trajectories 2: ", length(success_trajectories_2))
println("Success trajectories 3: ", length(success_trajectories_3))
println("Success trajectories: ", length(success_trajectories))
println("Failed trajectories: ", length(failed_trajectories))
#plot some trajectories
S_arr_unpack = unpack.(S_arr)
p1 = plot()
for i = 1:min(length(S_arr_unpack), 50)
    curr_state=S_arr_unpack[i].u_dict
    plot_ctmc!(p1, S_arr_unpack[i].times, curr_state[1], S_arr_unpack[i].final_time, c=:blue, linewidth=0.2)
    plot_ctmc!(p1, S_arr_unpack[i].times, curr_state[2], S_arr_unpack[i].final_time, c=:red, linewidth=0.2)
    plot_ctmc!(p1, S_arr_unpack[i].times, curr_state[3], S_arr_unpack[i].final_time, c=:green, linewidth=0.2)
end
display(p1)
savefig(p1, generate_filename(folder_name_for_plots, "multiple_trajectories_global.png"))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#now we will run the local optimizer
new_tau_opt_global=initial_tau_val_array#tau_opt
upper_bound_tau_0_local,upper_bound_local,tau_opt_local,P_opt_dict_local_temp,terminationstatus_local=run_nonlinear_solver(N, M_cell, λ, initial_state, Dict(1=>P1_opt_, 2=>P2_opt_, 3=>P3_opt_), new_tau_opt_global,Dict(1=>true,2=>true,3=>false),true, type_of_boundary_condition)


# Extract results
tau_opt_tilde_local = tau_opt_local[startstates]
lower_bound_local=minimum(tau_opt_tilde_local)

# Clean up P_opt values
P1_opt_local_ = P_opt_dict_local_temp[1] .* (abs.(P_opt_dict_local_temp[1]) .>= 1e-7)  # Zero out small values
P1_opt_local_ .= min.(P1_opt_local_, 1.0)                # Cap at 1.0
P2_opt_local_ = P_opt_dict_local_temp[2] .* (abs.(P_opt_dict_local_temp[2]) .>= 1e-7)  # Zero out small values
P2_opt_local_ .= min.(P2_opt_local_, 1.0)                # Cap at 1.0   
P3_opt_local_ = P_opt_dict_local_temp[3] .* (abs.(P_opt_dict_local_temp[3]) .>= 1e-7)  # Zero out small values
P3_opt_local_ .= min.(P3_opt_local_, 1.0)                # Cap at 1.0
Q_opt_local = Q_maker(Dict(1=>P1_opt_local_, 2=>P2_opt_local_, 3=>P3_opt_local_), N, M_cell, statedict, statedictinv)  # Generate Q matrix
Q_opt_absorbing_local=Q_absorbing_states_maker(Q_opt_local, all_targetstates)


#save the twocell_system 
parameters_opt_1_local = CellularDecisions_final.parameter_vector_to_parameters(P1_opt_local_, N)
parameters_opt_2_local = CellularDecisions_final.parameter_vector_to_parameters(P2_opt_local_, N)
parameters_opt_3_local = CellularDecisions_final.parameter_vector_to_parameters(P3_opt_local_, N)
parameters_opt_dict_local=Dict(1=>parameters_opt_1_local,2=>parameters_opt_2_local,3=>parameters_opt_3_local);

twocell_system_local = CellularDecisions_final.build_cell_system(N, M_cell, Q_opt_absorbing_local, parameters_opt_dict_local, type_of_boundary_condition)
# println(twocell_system.state_dict_inv)
twocell_system_filename_local = generate_filename(folder_name,"threecell_system_local")
CellularDecisions_final.save(twocell_system_local,twocell_system_filename_local)

#--------------------------------
#saving the optimal values
println("================================================")
hitting_mod_values_local=hitting_time_mod(Q_opt_absorbing_local,targetstates_good,targetstates_bad,startstates,λ)
#find the index of initial_state in startstates
initial_state_index=findfirst(x -> x == initial_state, startstates) 
println("hitting_time: ", hitting_mod_values_local[initial_state_index])
println("lambda: ", λ)
# Print to console
println("================================================")
println("Optimal solution")
println("P1_opt: ", P_opt_dict_local_temp[1])
println("P2_opt: ", P_opt_dict_local_temp[2])
println("P3_opt: ", P_opt_dict_local_temp[3])
println("tau_opt_0: ", tau_opt_local[initial_state])
println("------------------------------------------------")
println("F_1^+: ", parameters_opt_1_local.fp)
println("F_1^-: ", parameters_opt_1_local.fn)
println("F_2^+: ", parameters_opt_2_local.fp)
println("F_2^-: ", parameters_opt_2_local.fn)
println("F_3^+: ", parameters_opt_3_local.fp)
println("F_3^-: ", parameters_opt_3_local.fn)
println("G_1: ", parameters_opt_1_local.gp)
println("G_2: ", parameters_opt_2_local.gp)
println("G_3: ", parameters_opt_3_local.gp)
println("k_off_1: ", parameters_opt_1_local.koff)
println("k_off_2: ", parameters_opt_2_local.koff)
println("k_off_3: ", parameters_opt_3_local.koff)

# Save to txt file
optimal_values_filename_local = generate_filename(folder_name, "optimal_values_w_o_fixed_values_local.txt")
open(optimal_values_filename_local, "w") do io
    println(io, "tau_opt_0 = ", tau_opt_local[initial_state])
    println(io, "k_off_1 = ", P1_opt_local_[end])
    println(io, "F_1^+(.,0) = ", P1_opt_local_[1:N])
    println(io, "F_1^+(.,1) = ", P1_opt_local_[N+1:2*N])
    println(io, "F_1^-(.,0) = ", P1_opt_local_[2*N+1:3*N])
    println(io, "F_1^-(.,1) = ", P1_opt_local_[3*N+1:4*N])
    println(io, "G_1 = ", P1_opt_local_[4*N+1:end-1])
    println(io, "k_off_2 = ", P2_opt_local_[end])
    println(io, "F_2^+(.,0) = ", P2_opt_local_[1:N])
    println(io, "F_2^+(.,1) = ", P2_opt_local_[N+1:2*N])
    println(io, "F_2^-(.,0) = ", P2_opt_local_[2*N+1:3*N])
    println(io, "F_2^-(.,1) = ", P2_opt_local_[3*N+1:4*N])
    println(io, "G_2 = ", P2_opt_local_[4*N+1:end-1])
    println(io, "k_off_3 = ", P3_opt_local_[end])
    println(io, "F_3^+(.,0) = ", P3_opt_local_[1:N])
    println(io, "F_3^+(.,1) = ", P3_opt_local_[N+1:2*N])
    println(io, "F_3^-(.,0) = ", P3_opt_local_[2*N+1:3*N])
    println(io, "F_3^-(.,1) = ", P3_opt_local_[3*N+1:4*N])
    println(io, "G_3 = ", P3_opt_local_[4*N+1:end-1])
end

#--------------------------------
#plotting a few trajectories
T=25.0
num_simulations=1000

# Q_opt_absorbing=Q_absorbing_states_maker(Q, all_targetstates)
println("Simulating CTMC")
S_arr_local  = [CellularDecisions_final.simulate_ctmc(Q_opt_absorbing_local, initial_state, T,N, M_cell, type_of_boundary_condition) for i = 1:num_simulations]
println("Done simulating CTMC")
targetstates_good_1_local = targetstates_good[[statedict[t-1][2][1] ==0 && statedict[t-1][3][1] ==0  for t in targetstates_good]]
targetstates_good_2_local = targetstates_good[[statedict[t-1][1][1] ==0 && statedict[t-1][3][1] ==0  for t in targetstates_good]]
targetstates_good_3_local = targetstates_good[[statedict[t-1][1][1] ==0 && statedict[t-1][2][1] ==0  for t in targetstates_good]]

terminal_classes_local = [CellularDecisions_final.terminal_class(path, [targetstates_good_1_local,targetstates_good_2_local,targetstates_good_3_local], targetstates_bad) for path in S_arr_local]

success_trajectories_local = unpack.(S_arr_local[terminal_classes_local .!= -1])
failed_trajectories_local = unpack.(S_arr_local[terminal_classes_local .== -1])

failed_trajectories_local = (S_arr_local[terminal_classes_local .== -1])
success_trajectories_local = (S_arr_local[terminal_classes_local .== 1 .|| terminal_classes_local .== 2 .|| terminal_classes_local .== 3])
success_trajectories_1_local = (S_arr_local[terminal_classes_local .== 1])
success_trajectories_2_local = (S_arr_local[terminal_classes_local .== 2])
success_trajectories_3_local = (S_arr_local[terminal_classes_local .== 3])

println("Success trajectories 1: ", length(success_trajectories_1_local))
println("Success trajectories 2: ", length(success_trajectories_2_local))
println("Success trajectories 3: ", length(success_trajectories_3_local))
println("Success trajectories: ", length(success_trajectories_local))
println("Failed trajectories: ", length(failed_trajectories_local))
#plot some trajectories
S_arr_unpack_local = unpack.(S_arr_local)
p1 = plot()
for i = 1:min(length(S_arr_unpack_local), 50)
    curr_state=S_arr_unpack_local[i].u_dict
    plot_ctmc!(p1, S_arr_unpack_local[i].times, curr_state[1], S_arr_unpack_local[i].final_time, c=:blue, linewidth=0.2)
    plot_ctmc!(p1, S_arr_unpack_local[i].times, curr_state[2], S_arr_unpack_local[i].final_time, c=:red, linewidth=0.2)
    plot_ctmc!(p1, S_arr_unpack_local[i].times, curr_state[3], S_arr_unpack_local[i].final_time, c=:green, linewidth=0.2)
end
display(p1)
savefig(p1, generate_filename(folder_name_for_plots, "multiple_trajectories_local.png"))

println("Done")
println("================================================")
println("tau_opt_global: ", tau_opt[initial_state])
println("tau_opt_local: ", tau_opt_local[initial_state])
println("================================================")

