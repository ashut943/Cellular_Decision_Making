#This is an example optimization script for the three cell system, for the locally optimal solution.
#This is for a single initialization

using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO, LaTeXStrings
using Measures
using Revise
using CellularDecisions

include("mult_cell/mult_cell.jl")
include("utils/utils.jl")

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
function average_trajectory(path_array, time_points)
    u1_avg = zeros(length(time_points))
    u2_avg = zeros(length(time_points))
    u3_avg = zeros(length(time_points))
    u1_std = zeros(length(time_points))
    u2_std = zeros(length(time_points))
    u3_std = zeros(length(time_points))

    for (i,t) in enumerate(time_points)
        u1_pts = [path.u_dict[1][find_state_at_time(path,t)] for path in path_array]
        u2_pts = [path.u_dict[2][find_state_at_time(path,t)] for path in path_array]
        u3_pts = [path.u_dict[3][find_state_at_time(path,t)] for path in path_array]

        u1_avg[i] = mean(u1_pts)
        u2_avg[i] = mean(u2_pts)
        u3_avg[i] = mean(u3_pts)

        u1_std[i] = std(u1_pts)
        u2_std[i] = std(u2_pts)
        u3_std[i] = std(u3_pts)
    end
    return u1_avg, u2_avg, u3_avg, u1_std, u2_std, u3_std
end

function collect_transitions(path_array)
    N = path_array[1].internal_states
    all_transitions = vcat([[(path.states[i],path.states[i+1]) for i in 1:length(path.states)-1] for path in path_array]...)

    statedict, _, _, _, _, _ = CellularDecisions.statematrices(N,3, "boundary_2")
    u1_transitions = []
    u2_transitions = []
    u3_transitions = []
    for transition in all_transitions
        u_1_0 = statedict[transition[1] - 1][1][1]
        s_1_0 = statedict[transition[1] - 1][1][2]
        u_2_0 = statedict[transition[1] - 1][2][1]
        s_2_0 = statedict[transition[1] - 1][2][2]
        u_3_0 = statedict[transition[1] - 1][3][1]
        s_3_0 = statedict[transition[1] - 1][3][2]

        u_1_1 = statedict[transition[2] - 1][1][1]
        s_1_1 = statedict[transition[2] - 1][1][2]
        u_2_1 = statedict[transition[2] - 1][2][1]
        s_2_1 = statedict[transition[2] - 1][2][2]
        u_3_1 = statedict[transition[2] - 1][3][1]
        s_3_1 = statedict[transition[2] - 1][3][2]

        if (s_1_0 == s_1_1) && (u_1_0 == u_1_1) && (s_2_0 == s_2_1) && (u_2_0 == u_2_1)
            push!(u3_transitions, ([u_3_0,s_3_0], [u_3_1,s_3_1]))
        elseif (s_1_0 == s_1_1) && (u_1_0 == u_1_1) && (s_3_0 == s_3_1) && (u_3_0 == u_3_1)
            push!(u2_transitions, ([u_2_0,s_2_0], [u_2_1,s_2_1]))
        else
            push!(u1_transitions, ([u_1_0,s_1_0], [u_1_1,s_1_1]))
        end
    end

    transition_stats_1 = countmap(u1_transitions)
    transition_stats_2 = countmap(u2_transitions)
    transition_stats_3 = countmap(u3_transitions)
    return transition_stats_1, transition_stats_2, transition_stats_3, length(all_transitions)
end

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------

N = 10 # Number of states - 1
M_cell = 3  # Number of cells
K = 3  # Number of different P vectors to optimize (one for each cell type)
rho = [1, 1, 2]  # Cell 1 uses strategy 1, Cell 2 uses strategy 2, Cell 3 uses strategy 3
h_error = 0.02 # Target error rate
initial_tau_val =20.0 # Initial tau value for optimization
initial_state_array = ((1,0),(1,0),(1,0))  # Initial state for simulations
type_of_boundary_condition = "boundary_2"  # refer to boundary condition discussion elsewhere in the codebase


#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Create output directory
error_str = replace(string(round(h_error*100, digits=1)), "." => "_")
base_folder = joinpath(dirname(dirname(dirname(@__DIR__))), "results", "three_cell_results", "Interior_point_method_results_"*type_of_boundary_condition)
folder_name = joinpath(base_folder, @sprintf("Interior_Point_Method_results_N_%d_error_fix_%s", N, error_str))
mkpath(folder_name)

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Get state matrices, sizes, and target states
statedict,statedictinv,terminal_states,TG,TB,Tc=CellularDecisions.statematrices(N, M_cell, type_of_boundary_condition);
ni,np=CellularDecisions.varioussizes(N, M_cell);
ns=length(Tc);
targetstates_good=[target_state+1 for target_state ∈ TG];  # Good target states
targetstates_bad=[target_state+1 for target_state ∈ TB];   # Bad target states
targetstates=[targetstates_good;targetstates_bad];          # All target states
startstates=[start_state+1 for start_state ∈ Tc];         # Starting states
allstates=[startstates;targetstates_good; targetstates_bad];
all_targetstates = vcat(targetstates_good, targetstates_bad);

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
#initializing the initial value of the parameters (the strategy vector) for the optimization
initial_k_off_array = 0.34236413188666603
initial_f_plus_array_0 = [0.0, 0.1762932697050773, 0.9999999890267278, 0.4769223612336262, 0.6446712533120578, 0.8569258893992869]
initial_f_plus_array_1 = [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0]
initial_f_minus_array_0 = [0.0, 1.0, -0.0, -0.0, -0.0, 0.0]
initial_f_minus_array_1 = [1.0, 0.4563432292332993, 1.0, 1.0, 0.9999999989203174, -0.0]
initial_g_array = [-0.0, -0.0, -0.0, 1.0, 1.0, 1.0, 1.0]
#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------

initial_P_val_array = vcat(initial_f_plus_array_0, initial_f_plus_array_1, initial_f_minus_array_0, initial_f_minus_array_1, initial_g_array, initial_k_off_array)
initial_tau_val_array=[initial_tau_val for i in 1:ni]
for i in 1:ni
    if i ∈ targetstates_good || i ∈ targetstates_bad
        initial_tau_val_array[i]=0.0
    end
end
initial_state=statedictinv[initial_state_array]+1



#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
#find the tau_opt for the initial_state
Q_initial = Q_maker(Dict(1=>initial_P_val_array, 2=>initial_P_val_array, 3=>initial_P_val_array), N, M_cell, statedict, statedictinv)  # Generate Q matrix
Q_initial_absorbing=Q_absorbing_states_maker(Q_initial, all_targetstates)
initial_tau_opt=hitting_time_mod(Q_initial_absorbing,targetstates_good,targetstates_bad,startstates,0.0)
initial_h_error=hitting_prob_mod(Q_initial_absorbing,targetstates_good,targetstates_bad,startstates,0.0)

println("initial_tau_opt: ", initial_tau_opt[initial_state])
println("initial_h_error: ", initial_h_error[initial_state])
println("--------------------------------")

# Run nonlinear solver to get optimal solution
intial_tau_to_use=initial_tau_opt
upper_bound_tau_0, upper_bound, tau_opt, P_opt_dict_global_temp, terminationstatus = run_nonlinear_solver_modified_upper_bound_speed(N, M_cell, K, max_time, rho, h_error, initial_state, Dict(1=>initial_P_val_array, 2=>initial_P_val_array, 3=>initial_P_val_array), intial_tau_to_use, Dict(1=>true, 2=>true, 3=>false), type_of_boundary_condition)

if terminationstatus == MOI.LOCALLY_INFEASIBLE || terminationstatus == MOI.INFEASIBLE || terminationstatus == MOI.NUMERICAL_ERROR || terminationstatus == MOI.OTHER_ERROR || terminationstatus == MOI.ITERATION_LIMIT || terminationstatus == MOI.INTERRUPTED
    println("================================================")
    error("~~~~> Local optimization failed with status: $terminationstatus")
    println("================================================")
end
#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
# Clean up P_opt values, taking into account the floating point arithmetic errors
P1_opt_ = P_opt_dict_global_temp[1] .* (abs.(P_opt_dict_global_temp[1]) .>= 1e-7)  # Zero out small values
P1_opt_ .= min.(P1_opt_, 1.0)  # Cap at 1.0
P2_opt_ = P_opt_dict_global_temp[2] .* (abs.(P_opt_dict_global_temp[2]) .>= 1e-7)  # Zero out small values
P2_opt_ .= min.(P2_opt_, 1.0)  # Cap at 1.0
P3_opt_ = P_opt_dict_global_temp[3] .* (abs.(P_opt_dict_global_temp[3]) .>= 1e-7)  # Zero out small values
P3_opt_ .= min.(P3_opt_, 1.0)  # Cap at 1.0

# Create dictionaries for different cell type combinations
P_opt_dict = Dict(1=>P1_opt_, 2=>P2_opt_, 3=>P3_opt_)  # Both cells use strategy 1

Q_opt = Q_maker(P_opt_dict, N, M_cell, statedict, statedictinv)  # Generate Q matrix
Q_opt_absorbing=Q_absorbing_states_maker(Q_opt, all_targetstates)

#save the threecell_system 
parameters_opt_1 = CellularDecisions.parameter_vector_to_parameters(P1_opt_, N)
parameters_opt_2 = CellularDecisions.parameter_vector_to_parameters(P2_opt_, N)
parameters_opt_3 = CellularDecisions.parameter_vector_to_parameters(P3_opt_, N)
parameters_opt_dict=Dict(1=>parameters_opt_1,2=>parameters_opt_2,3=>parameters_opt_3);

threecell_system = CellularDecisions.build_cell_system(N, M_cell, Q_opt_absorbing, parameters_opt_dict, type_of_boundary_condition)
threecell_system_filename = generate_filename(folder_name,"threecell_system_local")
CellularDecisions.save(threecell_system,threecell_system_filename)

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------

#display the optimized values
hitting_times_expected=hitting_time_mod(Q_opt_absorbing,targetstates_good,targetstates_bad,startstates,0.0)
h_global_expected=hitting_prob_mod(Q_opt_absorbing,targetstates_good,targetstates_bad,startstates,0.0)

#printing the optimized solution
println("expected hitting time values: ", hitting_times_expected[initial_state]) 
println("expected hitting prob: ", h_global_expected[initial_state]) 
println("h_error: ", h_error)
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
println("------------------------------------------------")
println("expected hitting time values: ", hitting_times_expected[initial_state]) 
println("expected hitting prob: ", h_global_expected[initial_state]) 
println("h_error: ", h_error)

#saving the optimal values in txt
optimal_values_filename = generate_filename(folder_name, "optimal_values_w_o_fixed_values_local.txt")
open(optimal_values_filename, "w") do io
    println(io, "tau_opt_0 = ", tau_opt[initial_state])
    println(io, "h_error = ", h_error)
    println(io, "h_global = ", h_global[initial_state])
    println(io, "hitting_values = ", hitting_values[initial_state])
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

# #--------------------------------
# #++++++++++++++++++++++++++++++++
# #--------------------------------
# # MISCELLANEOUS CODE, CAN BE USED FOR PLOTTING         
# #--------------------------------
# #++++++++++++++++++++++++++++++++
# #--------------------------------

#Simulating the CTMC
T=50.0
num_simulations=1000

println("Simulating CTMC")
S_arr  = [CellularDecisions.simulate_ctmc(Q_opt_absorbing, initial_state, T,N, M_cell, type_of_boundary_condition) for i = 1:num_simulations]
println("Done simulating CTMC")
targetstates_good_1 = targetstates_good[[statedict[t-1][2][1] ==0 && statedict[t-1][3][1] ==0  for t in targetstates_good]]
targetstates_good_2 = targetstates_good[[statedict[t-1][1][1] ==0 && statedict[t-1][3][1] ==0  for t in targetstates_good]]
targetstates_good_3 = targetstates_good[[statedict[t-1][1][1] ==0 && statedict[t-1][2][1] ==0  for t in targetstates_good]]

terminal_classes = [CellularDecisions.terminal_class(path, [targetstates_good_1,targetstates_good_2,targetstates_good_3], targetstates_bad) for path in S_arr]

success_trajectories = unpack.(S_arr[terminal_classes .!= -1])
failed_trajectories = unpack.(S_arr[terminal_classes .== -1])

failed_trajectories = unpack.(S_arr[terminal_classes .== -1])
success_trajectories = unpack.(S_arr[terminal_classes .== 1 .|| terminal_classes .== 2 .|| terminal_classes .== 3])
success_trajectories_1 = unpack.(S_arr[terminal_classes .== 1])
success_trajectories_2 = unpack.(S_arr[terminal_classes .== 2])
success_trajectories_3 = unpack.(S_arr[terminal_classes .== 3])

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------

# Calculate proportions of each outcome
total = length(S_arr)
prop_failed = length(failed_trajectories) / total
prop_succ1 = length(success_trajectories_1) / total
prop_succ2 = length(success_trajectories_2) / total 
prop_succ3 = length(success_trajectories_3) / total
# Select representative trajectories based on proportions
num_failed = round(Int, 5 * prop_failed)
num_succ1 = round(Int, 5 * prop_succ1)
num_succ2 = round(Int, 5 * prop_succ2)
num_succ3 = max(round(Int, 5 * prop_succ3),1)
