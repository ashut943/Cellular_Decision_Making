#Example script for calculating the information metrics for the three cell system, in particular between the cells 3 and the tuple of cell 1 and 2 (together) (u3 and (u1,u2))

using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO
using Revise
using Plots.PlotMeasures
using CellularDecisions
using NumericalIntegration
using Interpolations
using Distributed
using SharedArrays
using JLD2

include("../../../mult_cell/mult_cell.jl")
include("../../../utils/utils.jl")
include("../../../information_metrics/infotheoryfuncs.jl")
include("../../../information_metrics/threecell_infotheory_calcs_two_v_one.jl")

N = 6  # Number of states- 1
T = 50.0  # Time for simulations
num_simulations = 10000  # N number of simulations
num_timesteps = 100 # Number of timesteps for the information metrics, i.e number of queries
initial_state_array = ((1,0),(1,0),(1,0))  # Initial state for simulations
type_of_boundary_condition="boundary_2" #type of boundary condition. This is the boundary condition of the system used in the papaer
type_of_optimisation="global" #as opposed to local optimal solution (which is the "greedy" solution in the paper)

h_error_array = [0.02]
#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------

final_mutual_information_array_two_v_one=[]
final_mutual_information_array_std_two_v_one=[]
final_transfer_entropy_array_xy_two_v_one=[]
final_transfer_entropy_array_xy_std_two_v_one=[]
final_transfer_entropy_array_yx_two_v_one=[]
final_transfer_entropy_array_yx_std_two_v_one=[]


for h_error in h_error_array
    println("h_error: ", h_error)
    println("--------------------------------")

    # Create output directory
    error_str = replace(string(round(h_error*100, digits=1)), "." => "_")
    base_folder = joinpath(dirname(dirname(dirname(@__DIR__))), "results", "three_cell_results", "Interior_point_method_results_"*type_of_boundary_condition)
    folder_name = joinpath(base_folder, @sprintf("Interior_Point_Method_results_low_info_N_%d_error_fix_%s", N, error_str))
    threecell_system_filename = generate_filename(folder_name,"threecell_system_"*type_of_optimisation)
    threecell_system = CellularDecisions.load(threecell_system_filename)

    plots_folder = joinpath(dirname(dirname(dirname(@__DIR__))), "plots", "three_cell_results", type_of_boundary_condition, "N_$(N)_error_fix_$(error_str)_low_info_$(type_of_optimisation)", "$(num_simulations)_$(num_timesteps)_two_v_one")
    mkpath(plots_folder)

    #--------------------------------
    #++++++++++++++++++++++++++++++++
    #--------------------------------

    # Get state matrices, sizes, and target states
    statedict=threecell_system.state_dict
    statedictinv=threecell_system.state_dict_inv
    terminal_states=threecell_system.terminal_states
    TG=threecell_system.T_good
    TB=threecell_system.T_bad
    Tc=threecell_system.Tc
    M_cell=threecell_system.num_cells
    ni,np=CellularDecisions.varioussizes(N,M_cell)
    ns=length(Tc);
    targetstates_good=[target_state+1 for target_state ∈ TG];  # Good target states
    targetstates_bad=[target_state+1 for target_state ∈ TB];   # Bad target states
    targetstates=[targetstates_good;targetstates_bad]          # All target states
    startstates=[start_state+1 for start_state ∈ Tc];         # Starting states
    allstates=[startstates;targetstates_good; targetstates_bad]
    all_targetstates = vcat(targetstates_good, targetstates_bad)
    initial_state=statedictinv[initial_state_array]+1

    #--------------------------------
    #++++++++++++++++++++++++++++++++
    #--------------------------------

    results_dict_global=calculate_information_metrics_multiple_trajectories(threecell_system, initial_state_array, T=T, initial_state=initial_state, num_simulations=num_simulations, num_timesteps=num_timesteps)

    #now calculate the overall metrics
    dict_info_unconditioned_global=calc_overall_info_metrics(results_dict_global, num_simulations)

    #save the dictionaries to a file
    @save folder_name*"/results_dict_two_v_one.jld2" results_dict_global
    @save folder_name*"/dict_info_unconditioned_two_v_one.jld2" dict_info_unconditioned_global

    #--------------------------------
    #++++++++++++++++++++++++++++++++
    #--------------------------------

    # plotting the results
    file_name=string(N)*"_"*string(h_error)*"_"*type_of_optimisation*"_"*string("unconditioned")
    plot_info_metrics(results_dict_global, dict_info_unconditioned_global, file_name, plots_folder)

    #--------------------------------
    #++++++++++++++++++++++++++++++++
    #--------------------------------

    #appending the results to the arrays
    mutual_info_curr=dict_info_unconditioned_global["avg_mutual_information"]
    mutual_info_std_curr=dict_info_unconditioned_global["std_error_mutual_information"]
    transfer_entropy_xy_curr=dict_info_unconditioned_global["avg_transfer_entropy_XY"]
    transfer_entropy_xy_std_curr=dict_info_unconditioned_global["std_error_XY"]
    transfer_entropy_yx_curr=dict_info_unconditioned_global["avg_transfer_entropy_YX"]
    transfer_entropy_yx_std_curr=dict_info_unconditioned_global["std_error_YX"]

    #finding the final time point values for the metrics
    final_mutual_info_curr=mutual_info_curr[end]
    final_mutual_info_std_curr=mutual_info_std_curr[end]
    final_transfer_entropy_xy_curr=transfer_entropy_xy_curr[end]
    final_transfer_entropy_xy_std_curr=transfer_entropy_xy_std_curr[end]
    final_transfer_entropy_yx_curr=transfer_entropy_yx_curr[end]
    final_transfer_entropy_yx_std_curr=transfer_entropy_yx_std_curr[end]

    #printing the results
    println("final_mutual_info_curr: ", final_mutual_info_curr)
    println("final_mutual_info_std_curr: ", final_mutual_info_std_curr)
    println("final_transfer_entropy_xy_curr: ", final_transfer_entropy_xy_curr)
    println("final_transfer_entropy_xy_std_curr: ", final_transfer_entropy_xy_std_curr)
    println("final_transfer_entropy_yx_curr: ", final_transfer_entropy_yx_curr)
    println("final_transfer_entropy_yx_std_curr: ", final_transfer_entropy_yx_std_curr)

    #appending the results to the arrays
    push!(final_mutual_information_array_two_v_one, final_mutual_info_curr)
    push!(final_mutual_information_array_std_two_v_one, final_mutual_info_std_curr)
    push!(final_transfer_entropy_array_xy_two_v_one, final_transfer_entropy_xy_curr)
    push!(final_transfer_entropy_array_xy_std_two_v_one, final_transfer_entropy_xy_std_curr)
    push!(final_transfer_entropy_array_yx_two_v_one, final_transfer_entropy_yx_curr)
    push!(final_transfer_entropy_array_yx_std_two_v_one, final_transfer_entropy_yx_std_curr)
end
