#This is an example script for sampling the error-speed phase space for the three cell system
#This is used to save and generate the randomly sampled points for the plot in the paper
#This is a smaller version, as one can use run this script in parallel (accordingly adjusting the file management) on a cluster

using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO, LaTeXStrings
using Measures
using Revise
using CellularDecisions
using Distributed

include("../../../mult_cell/mult_cell.jl")
include("../../../utils/utils.jl")

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
N = 6
M_cell = 3
K = 3
rho = [1, 1, 1]
initial_state_array = ((1,0),(1,0),(1,0))
type_of_boundary_condition = "boundary_2"


h_error = 0.02
initial_tau_val = 15.0

max_iter_number = 200   
N_sample=100
#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
statedict,statedictinv,terminal_states,TG,TB,Tc=CellularDecisions.statematrices(N, M_cell, type_of_boundary_condition);
ni,np=CellularDecisions.varioussizes(N, M_cell);
ns=length(Tc);
targetstates_good=[target_state+1 for target_state ∈ TG];  
targetstates_bad=[target_state+1 for target_state ∈ TB];  
targetstates=[targetstates_good;targetstates_bad];         
startstates=[start_state+1 for start_state ∈ Tc];        
allstates=[startstates;targetstates_good; targetstates_bad];
all_targetstates = vcat(targetstates_good, targetstates_bad);

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------

num_workers = Threads.nthreads()
println("Number of workers: ", num_workers)

# Hill function parameters (placeholder values)
# Here the hill function is:
# g(x) = 1 / ((K_hill / x)^n + 1)
# where x = 0:N

K_hill_array = [1.5,2.5,3.5,4.5]
min_params_array = [5.0*i for i in 1:20]


verbose_flag = false

error_rate_array = zeros(length(K_hill_array), length(min_params_array), N_sample)
speed_array = zeros(length(K_hill_array), length(min_params_array), N_sample)

for (K_hill_idx, K_hill) in enumerate(K_hill_array)
    for (min_params_idx, min_params) in enumerate(min_params_array)

        println("================================================")
        println("K_hill: ", K_hill, " min_params: ", min_params)
        println("================================================")

        error_rate_temp = zeros(N_sample)
        speed_temp = zeros(N_sample)

        Threads.@threads for sample in 1:N_sample
            initial_f_plus_array_0 = rand(N)
            initial_f_minus_array_0 = rand(N) 
            initial_f_plus_array_1 = rand(N)
            initial_f_minus_array_1 = rand(N)
            x = [i for i in 0:N]
            initial_g_array = @. 1 / ((K_hill/x)^n + 1)

            initial_k_off_array = [rand()]

            initial_P_val_array=vcat(initial_f_plus_array_0, initial_f_plus_array_1, initial_f_minus_array_0, initial_f_minus_array_1, initial_g_array, initial_k_off_array)
            initial_tau_val_array=[initial_tau_val for i in 1:ni]

            for i in 1:ni
                if i ∈ targetstates_good || i ∈ targetstates_bad
                    initial_tau_val_array[i]=0.0
                end
            end

            initial_state=statedictinv[initial_state_array]+1
            intial_tau_to_use=initial_tau_val_array
            tau_opt, P_opt_dict_global_temp, terminationstatus = run_nonlinear_solver_modified_for_search_in_phase_space(N, M_cell, K, rho, min_params, initial_state, Dict(1=>initial_P_val_array, 2=>initial_P_val_array, 3=>initial_P_val_array), intial_tau_to_use, Dict(1=>false, 2=>false, 3=>false), type_of_boundary_condition, max_iter_number, verbose_flag)

            P1_opt_ = P_opt_dict_global_temp[1]
            P2_opt_ = P_opt_dict_global_temp[2] 
            P3_opt_ = P_opt_dict_global_temp[3] 

            P_opt_dict = Dict(1=>P1_opt_, 2=>P2_opt_, 3=>P3_opt_)

            Q_opt = Q_maker(P_opt_dict, N, M_cell, statedict, statedictinv) 
            Q_opt_absorbing=Q_absorbing_states_maker(Q_opt, all_targetstates)

            hitting_times_expected = hitting_time_mod_inf_ok(Q_opt_absorbing, targetstates_good, targetstates_bad, startstates, 0.0)
            if Inf in hitting_times_expected
                println("Inf in hitting_times_expected")
                error_rate_temp[sample] = -1.0
                speed_temp[sample] = -1.0
                continue
            end
            hitting_prob_expected = hitting_prob_mod(Q_opt_absorbing, targetstates_good, targetstates_bad, startstates, 0.0)
            
            println("================================================")
            println("expected hitting time values: ", hitting_times_expected[initial_state]) 
            println("expected hitting prob: ", hitting_prob_expected[initial_state]) 
            println("================================================")

            error_rate_temp[sample] = hitting_prob_expected[initial_state]
            speed_temp[sample] = hitting_times_expected[initial_state]
        end
        error_rate_array[K_hill_idx, min_params_idx, :] = error_rate_temp
        speed_array[K_hill_idx, min_params_idx, :] = speed_temp  
        
    end
end

base_folder = joinpath(dirname(dirname(dirname(@__DIR__))), "results", "three_cell_results", "Interior_point_method_results_"*type_of_boundary_condition)
folder_name = joinpath(base_folder, @sprintf("sampled_phase_space"))
mkpath(folder_name)

save(joinpath(folder_name, "error_rate_array.jld2"), "error_rate_array", error_rate_array)
save(joinpath(folder_name, "speed_array.jld2"), "speed_array", speed_array)