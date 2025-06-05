# Import required packages
using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO
using Revise
using Plots.PlotMeasures
using CellularDecisions_final
using NumericalIntegration
using Interpolations
using Distributed
using SharedArrays
using JLD2



# Include helper files
include("../mult_cell/mult_cell_setup.jl")
include("../mult_cell/mult_cell_nonlinear.jl")
include("../mult_cell/mult_cell_hittingtime.jl")
include("../utils/ctmc_vis.jl")
include("../utils/ctmc_core.jl")
include("../utils/file_utils.jl")
include("../utils/video_utils.jl")
include("../information_metrics/infotheoryfuncs_two_cells.jl")
include("../information_metrics/twocell_infotheory_calcs.jl")

#--------------------------------
# Example usage
#--------------------------------
# load the data
N = 3  # Number of states- 1
λ = 30.0  # Lambda parameter
T = 50.0  # Time for simulations
num_simulations = 1000  # N number of simulations
num_timesteps = 1000 # Number of timesteps for the information metrics, i.e number of queries
initial_state_array = ((1,0),(1,0))  # Initial state for simulations
type_of_boundary_str="boundary_2"#for saving the results
type_of_boundary_condition="boundary_2"
type_of_optimisation_1="global"
type_of_optimisation_2="local"
#--------------------------------
# Create output directory
lambda_str = replace(string(λ), "." => "_")
base_folder = joinpath(dirname(@__DIR__), "experiments", "two_cell_results", "Interior_point_method_results_"*type_of_boundary_condition)
folder_name = joinpath(base_folder, @sprintf("Interior_Point_Method_results_N_%d_lambda_%s", N, lambda_str))
twocell_system_filename = generate_filename(folder_name,"twocell_system_"*type_of_optimisation_1)
twocell_system = CellularDecisions_final.load(twocell_system_filename)
plots_folder="./plots/two_cell_results/"*type_of_boundary_str*"/"*"N_$(N)_lambda_$(lambda_str)"*"_"*type_of_optimisation_1*"/"*string(num_simulations)*"/"
mkpath(plots_folder)
#--------------------------------
# Get state matrices, sizes, and target states
statedict=twocell_system.state_dict
statedictinv=twocell_system.state_dict_inv
terminal_states=twocell_system.terminal_states
TG=twocell_system.T_good
TB=twocell_system.T_bad
Tc=twocell_system.Tc
M_cell=twocell_system.num_cells
N=twocell_system.internal_states
ni,np=CellularDecisions_final.varioussizes(N,2)
ns=length(Tc);
targetstates_good=[target_state+1 for target_state ∈ TG];  # Good target states
targetstates_bad=[target_state+1 for target_state ∈ TB];   # Bad target states
targetstates=[targetstates_good;targetstates_bad]          # All target states
startstates=[start_state+1 for start_state ∈ Tc];         # Starting states
allstates=[startstates;targetstates_good; targetstates_bad]
all_targetstates = vcat(targetstates_good, targetstates_bad)
initial_state=statedictinv[initial_state_array]+1

#--------------------------------

results_dict_global=calculate_information_metrics_multiple_trajectories(twocell_system, initial_state_array, T=T, initial_state=initial_state, num_simulations=num_simulations, num_timesteps=num_timesteps)
# calculate_information_metrics(twocell_system, string(N)*"_"*string(λ), initial_state_array, T=T, initial_state=initial_state, num_simulations=num_simulations, plot_path=plots_folder)
#now calculate the overall metrics
dict_info_unconditioned_global, dict_info_conditioned_a_global, dict_info_conditioned_b_global, dict_info_conditioned_failed_global=calc_overall_info_metrics(results_dict_global, num_simulations)
#save the dictionaries to a file
@save plots_folder*"results_dict.jld2" results_dict_global
@save plots_folder*"dict_info_unconditioned.jld2" dict_info_unconditioned_global
@save plots_folder*"dict_info_conditioned_a.jld2" dict_info_conditioned_a_global
@save plots_folder*"dict_info_conditioned_b.jld2" dict_info_conditioned_b_global
@save plots_folder*"dict_info_conditioned_failed.jld2" dict_info_conditioned_failed_global

#--------------------------------
# plotting the results

file_name=string(N)*"_"*string(λ)*"_"*type_of_optimisation_1*"_"*string("unconditioned")
plot_info_metrics(results_dict_global, dict_info_unconditioned_global, file_name, plots_folder)

file_name=string(N)*"_"*string(λ)*"_"*type_of_optimisation_1*"_"*string("conditioned_a")
plot_info_metrics(results_dict_global, dict_info_conditioned_a_global, file_name, plots_folder)

file_name=string(N)*"_"*string(λ)*"_"*type_of_optimisation_1*"_"*string("conditioned_b")
plot_info_metrics(results_dict_global, dict_info_conditioned_b_global, file_name, plots_folder)

file_name=string(N)*"_"*string(λ)*"_"*type_of_optimisation_1*"_"*string("conditioned_failed")
plot_info_metrics(results_dict_global, dict_info_conditioned_failed_global, file_name, plots_folder)

#--------------------------------

#--------------------------------
#--------------------------------
#--------------------------------
#--------------------------------
#--------------------------------
#--------------------------------
#--------------------------------

#--------------------------------
# Create output directory
lambda_str = replace(string(λ), "." => "_")
base_folder = joinpath(dirname(@__DIR__), "experiments", "two_cell_results", "Interior_point_method_results_"*type_of_boundary_condition)
folder_name = joinpath(base_folder, @sprintf("Interior_Point_Method_results_N_%d_lambda_%s", N, lambda_str))
twocell_system_filename = generate_filename(folder_name,"twocell_system_"*type_of_optimisation_2)
twocell_system = CellularDecisions_final.load(twocell_system_filename)
plots_folder="./plots/two_cell_results/"*type_of_boundary_str*"/"*"N_$(N)_lambda_$(lambda_str)"*"_"*type_of_optimisation_2*"/"*string(num_simulations)*"/"
mkpath(plots_folder)
#--------------------------------
# Get state matrices, sizes, and target states
statedict=twocell_system.state_dict
statedictinv=twocell_system.state_dict_inv
terminal_states=twocell_system.terminal_states
TG=twocell_system.T_good
TB=twocell_system.T_bad
Tc=twocell_system.Tc
M_cell=twocell_system.num_cells
N=twocell_system.internal_states
ni,np=CellularDecisions_final.varioussizes(N,2)
ns=length(Tc);
targetstates_good=[target_state+1 for target_state ∈ TG];  # Good target states
targetstates_bad=[target_state+1 for target_state ∈ TB];   # Bad target states
targetstates=[targetstates_good;targetstates_bad]          # All target states
startstates=[start_state+1 for start_state ∈ Tc];         # Starting states
allstates=[startstates;targetstates_good; targetstates_bad]
all_targetstates = vcat(targetstates_good, targetstates_bad)
initial_state=statedictinv[initial_state_array]+1

#--------------------------------

results_dict_local=calculate_information_metrics_multiple_trajectories(twocell_system, initial_state_array, T=T, initial_state=initial_state, num_simulations=num_simulations, num_timesteps=num_timesteps)
# calculate_information_metrics(twocell_system, string(N)*"_"*string(λ), initial_state_array, T=T, initial_state=initial_state, num_simulations=num_simulations, plot_path=plots_folder)
#now calculate the overall metrics
dict_info_unconditioned_local, dict_info_conditioned_a_local, dict_info_conditioned_b_local, dict_info_conditioned_failed_local=calc_overall_info_metrics(results_dict_local, num_simulations)
#save the dictionaries to a file
@save plots_folder*"results_dict.jld2" results_dict_local
@save plots_folder*"dict_info_unconditioned.jld2" dict_info_unconditioned_local
@save plots_folder*"dict_info_conditioned_a.jld2" dict_info_conditioned_a_local
@save plots_folder*"dict_info_conditioned_b.jld2" dict_info_conditioned_b_local
@save plots_folder*"dict_info_conditioned_failed.jld2" dict_info_conditioned_failed_local

#--------------------------------
# plotting the results

file_name=string(N)*"_"*string(λ)*"_"*type_of_optimisation_2*"_"*string("unconditioned")
plot_info_metrics(results_dict_local, dict_info_unconditioned_local, file_name, plots_folder)

file_name=string(N)*"_"*string(λ)*"_"*type_of_optimisation_2*"_"*string("conditioned_a")
plot_info_metrics(results_dict_local, dict_info_conditioned_a_local, file_name, plots_folder)

file_name=string(N)*"_"*string(λ)*"_"*type_of_optimisation_2*"_"*string("conditioned_b")
plot_info_metrics(results_dict_local, dict_info_conditioned_b_local, file_name, plots_folder)

file_name=string(N)*"_"*string(λ)*"_"*type_of_optimisation_2*"_"*string("conditioned_failed")
plot_info_metrics(results_dict_local, dict_info_conditioned_failed_local, file_name, plots_folder)

#--------------------------------

#--------------------------------
#--------------------------------
#--------------------------------
#--------------------------------
#--------------------------------
#--------------------------------
#--------------------------------

#--------------------------------
#--------------------------------
function plot_local_global(ts_1,ts_2, dict_1, dict_2, type_of_optimisation_1, type_of_optimisation_2, path_combined, name_combined)
    plot_types = [
        "avg_mutual_information",
        "avg_transfer_entropy_XY",
        "avg_transfer_entropy_YX",
        "avg_mutual_information_rate_analytical",
        "avg_transfer_entropy_rate_XY_analytical",
        "avg_transfer_entropy_rate_YX_analytical",
    ]
    standard_errors=[
        "std_error_mutual_information",
        "std_error_XY",
        "std_error_YX",
        "std_error_mutual_information_rate_analytical",
        "std_error_transfer_entropy_rate_XY_analytical",
        "std_error_transfer_entropy_rate_YX_analytical",
    ]
    plot_titles = [
        "Mutual Information vs Time",
        "Transfer Entropy (Y→X) vs Time",
        "Transfer Entropy (X→Y) vs Time",
        "Mutual Information Rate vs Time",
        "Transfer Entropy Rate (Y→X) vs Time",
        "Transfer Entropy Rate (X→Y) vs Time",
    ]

    # Color palette for better visualization
    chosen_palette = :tab20


    # Generate each combined plot
    for (i, plot_type) in enumerate(plot_types)
        local p = plot(
            title = plot_titles[i],
            xlabel = "Time",
            ylabel = "Value",
            legend = :outertop,
            legend_columns = 2,
            dpi = 300,
            size = (900, 600),
            linewidth = 2,
            grid = false,
            background_color = :white,
            foreground_color = :black,
            margin = 10mm,
            palette = chosen_palette  
        )
        curr_standard_error_1=dict_1[standard_errors[i]]
        curr_standard_error_2=dict_2[standard_errors[i]]
        plot!(p, ts_1, dict_1[plot_type], 
                label = type_of_optimisation_1, 
                linewidth = 2,
                ribbon = curr_standard_error_1,
                fillalpha = 0.1)
        plot!(p, ts_2, dict_2[plot_type], 
                label = type_of_optimisation_2, 
                linewidth = 2,
                ribbon = curr_standard_error_2,
                fillalpha = 0.1)
        # Debug prints
        println("Saving to directory: ", path_combined)
        println("Full path: ", joinpath(path_combined, name_combined*"_$(plot_type).png"))
        
        # Make sure directory exists
        mkpath(path_combined)
        
        # Save the combined plot
        savefig(p, joinpath(path_combined, name_combined*"_$(plot_type).png"))
        display(p)
    end
end
ts_global=results_dict_global["ts"]
ts_local=results_dict_local["ts"]

combined_plot_dir = "./plots/two_cell_results/boundary_comparison/"*type_of_optimisation_1*"_vs_"*type_of_optimisation_2*"_w_"*type_of_boundary_condition*"/"*"N_$(N)_lambda_$(lambda_str)"*"/"*string(num_simulations)*"/"
mkpath(combined_plot_dir)

name_combined="combined_parameter_sweep_unconditioned"
path_combined=joinpath(combined_plot_dir,name_combined)
plot_local_global(ts_global,ts_local, dict_info_unconditioned_global, dict_info_unconditioned_local, type_of_optimisation_1, type_of_optimisation_2, path_combined, name_combined)

name_combined_conditioned_a="combined_parameter_sweep_conditioned_a"
path_combined_conditioned_a=joinpath(combined_plot_dir,name_combined_conditioned_a)
plot_local_global(ts_global,ts_local, dict_info_conditioned_a_global, dict_info_conditioned_a_local, type_of_optimisation_1, type_of_optimisation_2, path_combined_conditioned_a, name_combined_conditioned_a)

name_combined_conditioned_b="combined_parameter_sweep_conditioned_b"
path_combined_conditioned_b=joinpath(combined_plot_dir,name_combined_conditioned_b)
plot_local_global(ts_global,ts_local, dict_info_conditioned_b_global, dict_info_conditioned_b_local, type_of_optimisation_1, type_of_optimisation_2, path_combined_conditioned_b, name_combined_conditioned_b)

name_combined_conditioned_failed="combined_parameter_sweep_conditioned_failed"
path_combined_conditioned_failed=joinpath(combined_plot_dir,name_combined_conditioned_failed)
plot_local_global(ts_global,ts_local, dict_info_conditioned_failed_global, dict_info_conditioned_failed_local, type_of_optimisation_1, type_of_optimisation_2, path_combined_conditioned_failed, name_combined_conditioned_failed)

