#This is for calculation of informatin metrics of the three cell system (1,2,3) between cells 3 and the the random variable tuple of (cell 1, cell 2)

using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO
using Revise
using Plots.PlotMeasures
using CellularDecisions
using NumericalIntegration
using Interpolations
using Distributed
using SharedArrays


include("../mult_cell/mult_cell.jl")
include("../utils/utils.jl")
include("../information_metrics/infotheoryfuncs.jl")
include("../three_cell_coarsegraining/three_cell_coarse_graining_two_v_one.jl")

function calculate_information_metrics_multiple_trajectories(threecell_system, starting_state_tuple; T=50.0, initial_state=1, num_simulations=100, num_timesteps=1000)
    # Simulates multiple CTMC trajectories and computes path-wise transfer entropies and mutual information for each

    N = threecell_system.internal_states
    Q = threecell_system.Q_matrix
    TG = threecell_system.T_good
    TB = threecell_system.T_bad
    Tc = threecell_system.Tc
    statedict=threecell_system.state_dict
    statedictinv=threecell_system.state_dict_inv 

    ni, np = CellularDecisions.varioussizes(N,3)
    ns=length(Tc);
    targetstates_good = [target_state+1 for target_state ∈ TG]
    targetstates_bad = [target_state+1 for target_state ∈ TB]
    targetstates = [targetstates_good; targetstates_bad]
    startstates = [start_state+1 for start_state ∈ Tc]
    allstates = [startstates; targetstates_good; targetstates_bad]
        
    println("Simulating trajectories")
    S_arr = [CellularDecisions.simulate_ctmc(Q, initial_state, T, N, 3, "boundary_2") for i = 1:num_simulations]
    println("Simulated trajectories")
    targetstates_good_1 = targetstates_good[[statedict[t-1][2][1] ==0 && statedict[t-1][3][1] ==0  for t in targetstates_good]]
    targetstates_good_2 = targetstates_good[[statedict[t-1][1][1] ==0 && statedict[t-1][3][1] ==0  for t in targetstates_good]]
    targetstates_good_3 = targetstates_good[[statedict[t-1][1][1] ==0 && statedict[t-1][2][1] ==0  for t in targetstates_good]]
    
    terminal_classes = [CellularDecisions.terminal_class(path, [targetstates_good_1,targetstates_good_2,targetstates_good_3], targetstates_bad) for path in S_arr]
    
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
    
    ts = vcat(collect(range(0, T, num_timesteps)))
    
    num_workers = Threads.nthreads()
    println("Number of workers: ", num_workers)
    
    final_transfer_entropy_XY = zeros(num_simulations, length(ts))
    final_transfer_entropy_YX = zeros(num_simulations, length(ts))
    final_mutual_information = zeros(num_simulations, length(ts))
    final_transfer_entropy_rate_XY_analytical = zeros(num_simulations, length(ts))
    final_transfer_entropy_rate_YX_analytical = zeros(num_simulations, length(ts))
    final_mutual_information_rate_analytical = zeros(num_simulations, length(ts))
    
    Threads.@threads for trajectory_num in 1:length(S_arr)
        trajectory = S_arr[trajectory_num]
        local ts_temp_, path_te_XY_temp, path_te_YX_temp, path_mi_temp, path_te_XY_rate_analytical_temp, path_te_YX_rate_analytical_temp, path_mi_rate_analytical_temp= calculate_information_metrics_single_trajectory(threecell_system, trajectory, ts, starting_state_tuple)
        ts_indices_in_temp = [findall(x -> x == t, ts_temp_)[1] for t in ts if t ∈ ts_temp_]

        path_te_XY = path_te_XY_temp[ts_indices_in_temp]
        path_te_YX = path_te_YX_temp[ts_indices_in_temp]
        path_mi = path_mi_temp[ts_indices_in_temp]
        path_te_XY_rate_analytical = path_te_XY_rate_analytical_temp[ts_indices_in_temp]
        path_te_YX_rate_analytical = path_te_YX_rate_analytical_temp[ts_indices_in_temp]
        path_mi_rate_analytical = path_mi_rate_analytical_temp[ts_indices_in_temp]

        final_transfer_entropy_XY[trajectory_num, :] = path_te_XY
        final_transfer_entropy_YX[trajectory_num, :] = path_te_YX
        final_mutual_information[trajectory_num, :] = path_mi
        final_transfer_entropy_rate_XY_analytical[trajectory_num, :] = path_te_XY_rate_analytical
        final_transfer_entropy_rate_YX_analytical[trajectory_num, :] = path_te_YX_rate_analytical
        final_mutual_information_rate_analytical[trajectory_num, :] = path_mi_rate_analytical
        Threads.lock(ReentrantLock()) do
            println("Processed trajectory: ", trajectory_num, 
                    "; XY_transfer_entropy: ", path_te_XY[end],
                    "; YX_transfer_entropy: ", path_te_YX[end],
                    "; Path Mutual Information: ", path_mi[end],
                    "; Path Mutual Information Rate minimum: ", minimum(path_mi_rate_analytical),
                    "; Path Mutual Information Rate maximum: ", maximum(path_mi_rate_analytical)
                    )
        end
    end    
    
    results_dict = Dict(
        "S_arr" => S_arr,
        "ts" => ts,
        "terminal_classes" => terminal_classes,
        "final_transfer_entropy_XY" => final_transfer_entropy_XY,
        "final_transfer_entropy_YX" => final_transfer_entropy_YX,
        "final_mutual_information" => final_mutual_information,
        "final_transfer_entropy_rate_XY_analytical" => final_transfer_entropy_rate_XY_analytical,
        "final_transfer_entropy_rate_YX_analytical" => final_transfer_entropy_rate_YX_analytical,
        "final_mutual_information_rate_analytical" => final_mutual_information_rate_analytical
    )
    return results_dict

end

function helper_func_cal_metrics_given_indices(indices, full_final_transfer_entropy_XY, full_final_transfer_entropy_YX, full_final_mutual_information, full_final_transfer_entropy_rate_XY_analytical, full_final_transfer_entropy_rate_YX_analytical, full_final_mutual_information_rate_analytical)
    # Computes mean, std, and standard error of information metrics for a given subset of trajectory indices
    
    num_simulations = length(indices)

    final_transfer_entropy_XY = full_final_transfer_entropy_XY[indices, :]
    final_transfer_entropy_YX = full_final_transfer_entropy_YX[indices, :]
    final_mutual_information = full_final_mutual_information[indices, :]
    final_transfer_entropy_rate_XY_analytical = full_final_transfer_entropy_rate_XY_analytical[indices, :]
    final_transfer_entropy_rate_YX_analytical = full_final_transfer_entropy_rate_YX_analytical[indices, :]
    final_mutual_information_rate_analytical = full_final_mutual_information_rate_analytical[indices, :]

    final_average_transfer_entropy_XY = vec(mean(final_transfer_entropy_XY, dims=1))
    final_average_transfer_entropy_YX = vec(mean(final_transfer_entropy_YX, dims=1))
    final_average_mutual_information = vec(mean(final_mutual_information, dims=1))
    final_average_transfer_entropy_rate_XY_analytical = vec(mean(final_transfer_entropy_rate_XY_analytical, dims=1))
    final_average_transfer_entropy_rate_YX_analytical = vec(mean(final_transfer_entropy_rate_YX_analytical, dims=1))
    final_average_mutual_information_rate_analytical = vec(mean(final_mutual_information_rate_analytical, dims=1))
    
    final_std_transfer_entropy_XY = vec(std(final_transfer_entropy_XY, dims=1))
    final_std_transfer_entropy_YX = vec(std(final_transfer_entropy_YX, dims=1))
    final_std_mutual_information = vec(std(final_mutual_information, dims=1))
    final_std_transfer_entropy_rate_XY_analytical = vec(std(final_transfer_entropy_rate_XY_analytical, dims=1))
    final_std_transfer_entropy_rate_YX_analytical = vec(std(final_transfer_entropy_rate_YX_analytical, dims=1))
    final_std_mutual_information_rate_analytical = vec(std(final_mutual_information_rate_analytical, dims=1))

    final_std_error_XY = final_std_transfer_entropy_XY ./ sqrt(num_simulations)
    final_std_error_YX = final_std_transfer_entropy_YX ./ sqrt(num_simulations)
    final_std_error_mutual_information = final_std_mutual_information ./ sqrt(num_simulations)
    final_std_error_transfer_entropy_rate_XY_analytical = final_std_transfer_entropy_rate_XY_analytical ./ sqrt(num_simulations)
    final_std_error_transfer_entropy_rate_YX_analytical = final_std_transfer_entropy_rate_YX_analytical ./ sqrt(num_simulations)
    final_std_error_mutual_information_rate_analytical = final_std_mutual_information_rate_analytical ./ sqrt(num_simulations)

    return Dict(
        "avg_transfer_entropy_XY" => final_average_transfer_entropy_XY,
        "avg_transfer_entropy_YX" => final_average_transfer_entropy_YX,
        "avg_mutual_information" => final_average_mutual_information,
        "avg_mutual_information_rate_analytical" => final_average_mutual_information_rate_analytical,
        "avg_transfer_entropy_rate_XY_analytical" => final_average_transfer_entropy_rate_XY_analytical,
        "avg_transfer_entropy_rate_YX_analytical" => final_average_transfer_entropy_rate_YX_analytical,
        "std_XY" => final_std_transfer_entropy_XY,
        "std_YX" => final_std_transfer_entropy_YX,
        "std_mutual_information" => final_std_mutual_information,
        "std_mutual_information_rate_analytical" => final_std_mutual_information_rate_analytical,
        "std_transfer_entropy_rate_XY_analytical" => final_std_transfer_entropy_rate_XY_analytical,
        "std_transfer_entropy_rate_YX_analytical" => final_std_transfer_entropy_rate_YX_analytical,
        "std_error_XY" => final_std_error_XY,
        "std_error_YX" => final_std_error_YX,
        "std_error_mutual_information" => final_std_error_mutual_information,
        "std_error_mutual_information_rate_analytical" => final_std_error_mutual_information_rate_analytical,
        "std_error_transfer_entropy_rate_XY_analytical" => final_std_error_transfer_entropy_rate_XY_analytical,
        "std_error_transfer_entropy_rate_YX_analytical" => final_std_error_transfer_entropy_rate_YX_analytical,
    )
end

function calc_overall_info_metrics(results_dict, num_simulations)
    # Filters out trajectories with extreme MI rates and returns aggregated statistics over the remaining ones
   terminal_classes=results_dict["terminal_classes"]
   final_transfer_entropy_XY=results_dict["final_transfer_entropy_XY"]
   final_transfer_entropy_YX=results_dict["final_transfer_entropy_YX"]
   final_mutual_information=results_dict["final_mutual_information"]
   final_transfer_entropy_rate_XY_analytical=results_dict["final_transfer_entropy_rate_XY_analytical"]
   final_transfer_entropy_rate_YX_analytical=results_dict["final_transfer_entropy_rate_YX_analytical"]
   final_mutual_information_rate_analytical=results_dict["final_mutual_information_rate_analytical"]
   
    all_indices = []
    for i in 1:num_simulations
        curr_mi_rate_analytical=final_mutual_information_rate_analytical[i,:]
        if maximum(abs.(curr_mi_rate_analytical)) < 1e1
            push!(all_indices, i)
        end
    end
    println("Number of trajectories: ", length(all_indices))
    dict_info_unconditioned = helper_func_cal_metrics_given_indices(all_indices, final_transfer_entropy_XY, final_transfer_entropy_YX, final_mutual_information, final_transfer_entropy_rate_XY_analytical, final_transfer_entropy_rate_YX_analytical, final_mutual_information_rate_analytical)

    return dict_info_unconditioned
end

function plot_info_metrics(results_dict, info_dict, file_name, plot_path)
    # Generates and saves plots of transfer entropies, mutual information, and their rates with confidence ribbons

    ts = results_dict["ts"]
    mkpath(plot_path)

    combined_plot_XY = plot(ts, info_dict["avg_transfer_entropy_XY"],
        ribbon=info_dict["std_XY"],
        label="Average",
        fillalpha=0.3,
        fillcolor=:blue,
        xlabel="Time",
        ylabel="Transfer Entropy", 
        title="Transfer Entropy Y→X Across Trajectories",
        lw=2,
        legend=false)   
    plot!(ts, info_dict["avg_transfer_entropy_XY"],
        ribbon=info_dict["std_error_XY"],
        fillalpha=0.5,
        fillcolor=:red)

    for i in 1:3
        plot!(ts, results_dict["final_transfer_entropy_XY"][i,:], 
            alpha=0.3,
            color=:blue,
            lw=1)
    end

    savefig(combined_plot_XY, joinpath(plot_path, "transfer_xy_"*file_name*".png"))
    savefig(combined_plot_XY, joinpath(plot_path, "transfer_xy_"*file_name*".svg"))


    combined_plot_XY_rate_for_single_path = plot(ts, info_dict["avg_transfer_entropy_rate_XY_analytical"],
        ribbon=info_dict["std_error_transfer_entropy_rate_XY_analytical"],
        fillalpha=0.5,
        fillcolor=:blue,
        xlabel="Time", 
        ylabel="Transfer Entropy Rate",
        title="Analytical Transfer Entropy Rate Y→X Across Trajectories",
        lw=2,
        legend=false)

        
        
    savefig(combined_plot_XY_rate_for_single_path, joinpath(plot_path, "transfer_rate_xy_analytical_"*file_name*".png"))
    savefig(combined_plot_XY_rate_for_single_path, joinpath(plot_path, "transfer_rate_xy_analytical_"*file_name*".svg"))

    combined_plot_YX_rate_for_single_path = plot(ts, info_dict["avg_transfer_entropy_rate_XY_analytical"],
        ribbon=info_dict["std_error_transfer_entropy_rate_XY_analytical"],
        fillalpha=0.5,
        fillcolor=:blue,
        xlabel="Time",
        ylabel="Transfer Entropy Rate",
        title="Analytical Transfer Entropy Rate Y→X Across Trajectories",
        lw=2,
        legend=false)

    plot!(ts[1:end-1], diff(info_dict["avg_transfer_entropy_XY"]) ./ diff(ts),
        fillalpha=0.5,
        fillcolor=:red)
    

    savefig(combined_plot_YX_rate_for_single_path, joinpath(plot_path, "transfer_rate_yx_analytical_"*file_name*".png"))
    savefig(combined_plot_YX_rate_for_single_path, joinpath(plot_path, "transfer_rate_yx_analytical_"*file_name*".svg"))


    combined_plot_YX = plot(ts, info_dict["avg_transfer_entropy_YX"],
        ribbon=info_dict["std_YX"],
        fillalpha=0.3,
        fillcolor=:blue,
        xlabel="Time",
        ylabel="Transfer Entropy", 
        title="Transfer Entropy X→Y Across Trajectories",
        lw=2,
        legend=false)

    plot!(ts, info_dict["avg_transfer_entropy_YX"],
        ribbon=info_dict["std_error_YX"],
        fillalpha=0.5,
        fillcolor=:red)

    for i in 1:3
        plot!(ts, results_dict["final_transfer_entropy_YX"][i,:],
            alpha=0.3,
            color=:blue,
            lw=1)
    end
        
    savefig(combined_plot_YX, joinpath(plot_path, "transfer_yx_"*file_name*".png"))
    savefig(combined_plot_YX, joinpath(plot_path, "transfer_yx_"*file_name*".svg"))

    combined_plot_YX_rate_for_single_path = plot(ts, info_dict["avg_transfer_entropy_rate_YX_analytical"],
        ribbon=info_dict["std_error_transfer_entropy_rate_YX_analytical"],
        fillalpha=0.3,
        fillcolor=:blue,
        xlabel="Time",
        ylabel="Transfer Entropy Rate", 
        title="Analytical Transfer Entropy Rate X→Y Across Trajectories",
        lw=2,
        legend=false)

    plot!(ts[1:end-1], diff(info_dict["avg_transfer_entropy_YX"]) ./ diff(ts),
        fillalpha=0.5,
        fillcolor=:red)

    savefig(combined_plot_YX_rate_for_single_path, joinpath(plot_path, "transfer_rate_yx_analytical_"*file_name*".png"))
    savefig(combined_plot_YX_rate_for_single_path, joinpath(plot_path, "transfer_rate_yx_analytical_"*file_name*".svg"))
    
    combined_plot_mutual_information = plot(ts, info_dict["avg_mutual_information"],
        ribbon=info_dict["std_mutual_information"],
        fillalpha=0.3,
        fillcolor=:blue,
        xlabel="Time",
        ylabel="Mutual Information", 
        title="Mutual Information Across Trajectories",
        lw=2,
        legend=false)

    plot!(ts, info_dict["avg_mutual_information"],
        ribbon=info_dict["std_error_mutual_information"],
        fillalpha=0.5,
        fillcolor=:red)

    for i in 1:3
        plot!(ts, results_dict["final_mutual_information"][i,:],
            alpha=0.3,
            color=:blue,
            lw=1)
    end
        
    savefig(combined_plot_mutual_information, joinpath(plot_path, "mi_"*file_name*".png"))
    savefig(combined_plot_mutual_information, joinpath(plot_path, "mi_"*file_name*".svg"))
    
    combined_plot_mutual_information_rate_for_single_path = plot(ts, info_dict["avg_mutual_information_rate_analytical"],
        ribbon=info_dict["std_error_mutual_information_rate_analytical"],
        fillalpha=0.5,
        fillcolor=:blue,
        xlabel="Time",
        ylabel="Mutual Information Rate",
        title="Analytical Mutual Information Rate Across Trajectories",
        lw=2,
        legend=false)

    plot!(ts[1:end-1], diff(info_dict["avg_mutual_information"]) ./ diff(ts),
        fillalpha=0.5,
        fillcolor=:red)

    savefig(combined_plot_mutual_information_rate_for_single_path, joinpath(plot_path, "mi_rate_single_path_"*file_name*".png"))
    savefig(combined_plot_mutual_information_rate_for_single_path, joinpath(plot_path, "mi_rate_single_path_"*file_name*".svg"))

    multiplier=1.96
    combined_plot_rates = plot(ts, info_dict["avg_mutual_information_rate_analytical"],
        ribbon=multiplier*info_dict["std_error_mutual_information_rate_analytical"],
        fillalpha=0.5,
        fillcolor=:purple,
        color=:purple,
        xlabel="Time",
        ylabel="Rate Value",
        title="Information Theory Rates",
        label="Mutual Information Rate",
        lw=2,
        legend=true)
    plot!(ts, info_dict["avg_transfer_entropy_rate_XY_analytical"],
    ribbon=multiplier*info_dict["std_error_transfer_entropy_rate_XY_analytical"],
    fillalpha=0.3,
    fillcolor=:green,
    color=:green,
    xlabel="Time",
    ylabel="Rate Value", 
    label="Transfer Entropy Rate Y→X",
    lw=2,
    legend=true)
    
    plot!(ts, info_dict["avg_transfer_entropy_rate_YX_analytical"],
    ribbon=multiplier*info_dict["std_error_transfer_entropy_rate_YX_analytical"],
    fillalpha=0.3,
    fillcolor=:brown,
    color=:brown,
    xlabel="Time",
    ylabel="Rate Value", 
    label="Transfer Entropy Rate X→Y",
    lw=2,
    legend=true)

    savefig(combined_plot_rates, joinpath(plot_path, "rates_combined_"*file_name*".png"))
    savefig(combined_plot_rates, joinpath(plot_path, "rates_combined_"*file_name*".svg"))

    combined_plot_integrals = plot(ts, info_dict["avg_mutual_information"],
    ribbon=multiplier*info_dict["std_error_mutual_information"],
    fillalpha=0.5,
    fillcolor=:purple,
    color=:purple,
    xlabel="Time",
    ylabel="Integral Value", 
    title="Information Theory Integrals",
    label="Mutual Information",
    lw=2,
    legend=true)
    plot!(ts, info_dict["avg_transfer_entropy_XY"],
    ribbon=multiplier*info_dict["std_error_XY"],
    fillalpha=0.3,
    fillcolor=:green,
    color=:green,
    xlabel="Time",
    ylabel="Transfer Entropy",
    label="Transfer Entropy Y→X",
    lw=2,
    legend=true)

    plot!(ts, info_dict["avg_transfer_entropy_YX"],
    ribbon=multiplier*info_dict["std_error_YX"],
    fillalpha=0.3,
    fillcolor=:brown,
    color=:brown,
    xlabel="Time",
    ylabel="Transfer Entropy",
    label="Transfer Entropy X→Y", 
    lw=2,
    legend=true)

    savefig(combined_plot_integrals, joinpath(plot_path, "integrals_combined_"*file_name*".png"))
    savefig(combined_plot_integrals, joinpath(plot_path, "integrals_combined_"*file_name*".svg"))
end