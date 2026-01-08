#This is an example script for getting the Doob transform CTMC's rate matrix, that is conditioned on the event of the system succesful patterning AND cell 3 expressing itself

using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO, LaTeXStrings
using Measures
using Revise
using CellularDecisions

include("../../../mult_cell/mult_cell.jl")
include("../../../utils/utils.jl")

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
        # ua_pts = [path.u1[find_state_at_time(path,t)] for path in path_array]
        # ub_pts = [path.u2[find_state_at_time(path,t)] for path in path_array]

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

N = 6 # Number of states - 1
M_cell = 3  # Number of cells
h_error = 0.10  # Target error rate
initial_state_array = ((1,0),(1,0),(1,0))  # Initial state for simulations
type_of_boundary_condition = "boundary_2"  # for saving the results

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------

# Create output directory
error_str = replace(string(round(h_error*100, digits=1)), "." => "_")
base_folder = joinpath(dirname(@__DIR__), "experiments", "three_cell_results", "Interior_point_method_results_"*type_of_boundary_condition)
folder_name = joinpath(base_folder, @sprintf("Interior_Point_Method_results_N_%d_error_fix_%s", N, error_str))

folder_name_for_plots = joinpath(dirname(dirname(dirname(@__DIR__))), "plots", "three_cell_results", type_of_boundary_condition, "N_$(N)_error_fix_$(error_str)_event_basis")
mkpath(folder_name_for_plots)

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------

# Get state matrices, sizes, and target states
statedict,statedictinv,terminal_states,TG,TB,Tc=CellularDecisions.statematrices(N, M_cell, type_of_boundary_condition);
ni,np=CellularDecisions.varioussizes(N, M_cell);
ns=length(Tc);
targetstates_good=[target_state+1 for target_state ∈ TG];
targetstates_bad=[target_state+1 for target_state ∈ TB];
targetstates=[targetstates_good;targetstates_bad];
startstates=[start_state+1 for start_state ∈ Tc];
allstates=[startstates;targetstates_good; targetstates_bad];
all_targetstates = vcat(targetstates_good, targetstates_bad);
initial_state=statedictinv[initial_state_array]+1

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------

# load the original system that was globally optimized and saved
threecell_system_filename = generate_filename(folder_name,"threecell_system_global")
threecell_system=CellularDecisions.load(threecell_system_filename)
Q_opt_absorbing=threecell_system.Q_matrix

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
#sanity check to display the optimal values
hitting_values=hitting_time_mod(Q_opt_absorbing,targetstates_good,targetstates_bad,startstates,0.0)
h_global=hitting_prob_mod(Q_opt_absorbing,targetstates_good,targetstates_bad,startstates,0.0)

#then find the expected hitting time and hitting probability
hitting_times_expected=hitting_values
h_global_expected=h_global
println("expected hitting time values: ", hitting_times_expected[initial_state]) 
println("expected hitting prob: ", h_global_expected[initial_state]) 
println("h_error: ", h_error)
println("------------------------------------------------")

#--------------------------------
#++++++++++++++++++++++++++++++++
#--------------------------------
#Now doob transformation

#Step 1: Define the event as being hitting the target state where cell 3 has won
new_good_states=[]
for target_states_new_idx in targetstates_good
    target_state_idx=target_states_new_idx-1
    target_state_now=statedict[target_state_idx]
    if target_state_now[3][1]==N
        push!(new_good_states,target_states_new_idx)
    end
end
new_bad_states=[]
for target_states_new_idx in targetstates
    target_state_idx=target_states_new_idx-1
    target_state_now=statedict[target_state_idx]
    if !(target_states_new_idx in new_good_states)
        push!(new_bad_states,target_states_new_idx)
    end
end

new_targetstates=[new_good_states;new_bad_states]
all_targetstates_new=new_targetstates
startstates_new=[startstates]
println("Length of new good states: ", length(new_good_states))
println("Length of new bad states: ", length(new_bad_states))
println("Length of original good states: ", length(targetstates_good))
println("Length of original bad states: ", length(targetstates_bad))

h_global_new=hitting_prob_mod(Q_opt_absorbing,new_bad_states,new_good_states,startstates_new,1.0)

println("expected hitting prob: ", h_global_new[initial_state]) 
println("h_error: ", h_error)
println("------------------------------------------------")

# Step 2: Need to find weights for the reweighting procedure
#define matrix omega of size Q_opt_absorbing
omega=zeros(size(Q_opt_absorbing))
for i in 1:size(Q_opt_absorbing,1)
    for j in 1:size(Q_opt_absorbing,2)
        denom=h_global_new[i]
        numer=h_global_new[j]
        omega[i,j]=0.0
        if denom==0.0
            if abs(numer)>1e-7
                omega[i,j]=Inf
            else
                omega[i,j]=1.0
            end
        else
            omega[i,j]=numer/denom
            if abs(omega[i,j])<1e-7
                omega[i,j]=0.0
            end
        end
    end
end
reweighted_Q=zeros(size(Q_opt_absorbing))
for i in 1:size(Q_opt_absorbing,1)
    for j in 1:size(Q_opt_absorbing,2)
        if omega[i,j]==Inf && Q_opt_absorbing[i,j]!=0.0
            error("omega[i,j] is Inf")
            #sanity check
        end
        if omega[i,j]==Inf
            reweighted_Q[i,j]=0.0
        else
            reweighted_Q[i,j]=Q_opt_absorbing[i,j]*omega[i,j]
        end
    end
end

#save the reweighted Q
h_global_new=hitting_prob_mod(reweighted_Q,new_bad_states,new_good_states,startstates_new,1.0)
hitting_values=hitting_time_mod_inf_ok(reweighted_Q,new_good_states,new_bad_states,startstates_new,0.0)
println("expected hitting time values: ", hitting_values[initial_state]) 
println("expected hitting prob: ", h_global_new[initial_state]) 
println("h_error: ", h_error)
println("------------------------------------------------")

#Now we have the reweighted Q

#create a new cell system that is a copy of the previous one, but change the Q_matrix to the reweighted Q
parameters_opt_dict_original=threecell_system.parameters_dict

threecell_system_new=CellularDecisions.build_cell_system(N, M_cell, reweighted_Q, parameters_opt_dict_original, type_of_boundary_condition)

#save the new cell system
threecell_system_new_filename = generate_filename(folder_name,"threecell_system_global_reweighted_cell_3_wins")
CellularDecisions.save(threecell_system_new, threecell_system_new_filename)

# #--------------------------------
# #++++++++++++++++++++++++++++++++
# #--------------------------------
# # MISCELLANEOUS CODE THAT IS PLOTTING THE TRAJECTORIES, AND SOME KYMOGRAPHS      
# #--------------------------------
# #++++++++++++++++++++++++++++++++
# #--------------------------------

#plotting a few trajectories
T=50.0
num_simulations=1000
Q_opt_absorbing=threecell_system_new.Q_matrix

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

# Ensure we get exactly 5 trajectories by adjusting failed category if needed
total_selected = num_failed + num_succ1 + num_succ2 + num_succ3
if total_selected < 5
    num_failed += 5 - total_selected
end

if total_selected > 5
    num_failed -= total_selected - 5
end

# Randomly sample from each category
representative_trajectories = vcat(
    length(failed_trajectories) > 0 ? rand(failed_trajectories, min(num_failed, length(failed_trajectories))) : [],
    length(success_trajectories_1) > 0 ? rand(success_trajectories_1, min(num_succ1, length(success_trajectories_1))) : [],
    length(success_trajectories_2) > 0 ? rand(success_trajectories_2, min(num_succ2, length(success_trajectories_2))) : [],
    length(success_trajectories_3) > 0 ? rand(success_trajectories_3, min(num_succ3, length(success_trajectories_3))) : []
)

all_trajectories = representative_trajectories


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
savefig(p1, generate_filename(folder_name_for_plots, "multiple_trajectories_global_reweighted_cell_3_wins.png"))
savefig(p1, generate_filename(folder_name_for_plots, "multiple_trajectories_global_reweighted_cell_3_wins.svg"))

t_plot = 0:0.5:50
u1_avg_fail, u2_avg_fail, u3_avg_fail, u1_std_fail, u2_std_fail, u3_std_fail = average_trajectory(failed_trajectories, t_plot)
u1_avg_succ, u2_avg_succ, u3_avg_succ, u1_std_succ, u2_std_succ, u3_std_succ = average_trajectory(success_trajectories_3, t_plot)
p1 = plot(t_plot,u1_avg_fail,ribbon = u1_std_fail,label="cell 1",xlabel="Time",ylabel="State",title="Unsuccesful trajectories", color=:purple)
plot!(p1,t_plot,u2_avg_fail,ribbon = u2_std_fail,label="cell 2", color=:green)
plot!(p1,t_plot,u3_avg_fail,ribbon = u3_std_fail,label="cell 3", color=:brown)
savefig(p1, generate_filename(folder_name_for_plots, "multiple_trajectories_global_reweighted_cell_3_wins_avg_fail.png")) 
savefig(p1, generate_filename(folder_name_for_plots, "multiple_trajectories_global_reweighted_cell_3_wins_avg_fail.svg"))

p2 = plot(t_plot,u1_avg_succ,ribbon = u1_std_succ,label="cell 1",xlabel="Time",ylabel="State",title="Succesful trajectories", color=:purple)
plot!(p2, t_plot,u2_avg_succ,ribbon = u2_std_succ,label="cell 2", color=:green)
plot!(p2, t_plot,u3_avg_succ,ribbon = u3_std_succ,label="cell 3", color=:brown)
savefig(p2, generate_filename(folder_name_for_plots, "multiple_trajectories_global_reweighted_cell_3_wins_avg_success.png"))
savefig(p2, generate_filename(folder_name_for_plots, "multiple_trajectories_global_reweighted_cell_3_wins_avg_success.svg"))
p3=plot(p1,p2,layout=(2,1))
display(p3)
savefig(p3, generate_filename(folder_name_for_plots, "multiple_trajectories_global_reweighted_cell_3_wins_avg_both_success_and_fail.png"))
savefig(p3, generate_filename(folder_name_for_plots, "multiple_trajectories_global_reweighted_cell_3_wins_avg_both_success_and_fail.svg"))


# Fixed version of the kymograph signaling status plotting
# Create kymograph for a single successful trajectory as a 2D image (true kymograph)
p_all=[]
num_indices_to_plot = 5

# Calculate global min/max for consistent color scale across all plots
global kymo_min_state = Inf
global kymo_max_state = -Inf


if length(all_trajectories) > 0
    for index_to_plot in 1:min(num_indices_to_plot, length(all_trajectories))
        traj = all_trajectories[index_to_plot]
        for cell in 1:3
            cell_states = traj.u_dict[cell]
            if !isempty(cell_states)
                global kymo_min_state = min(kymo_min_state, minimum(cell_states))
                global kymo_max_state = max(kymo_max_state, maximum(cell_states))
            end
        end
    end
end
println("kymo_min_state: ", kymo_min_state)
println("kymo_max_state: ", kymo_max_state)

if kymo_min_state == Inf || kymo_max_state == -Inf
    error("kymo_min_state or kymo_max_state is Inf or -Inf")
end

if length(all_trajectories) > 0
    println("Kymograph color scale: $kymo_min_state to $kymo_max_state")
    
    for index_to_plot in 1:num_indices_to_plot
        traj = all_trajectories[index_to_plot]
        # Find the last jump time for either cell
        last_jump_1 = traj.times[end]
        last_jump_2 = traj.times[end]
        if length(traj.u_dict[1]) > 1
            last_jump_1 = traj.times[findlast(diff(traj.u_dict[1]) .!= 0)]
        end
        if length(traj.u_dict[2]) > 1
            last_jump_2 = traj.times[findlast(diff(traj.u_dict[2]) .!= 0)]
        end
        last_jump = max(last_jump_1, last_jump_2)
        println("last_jump: ", last_jump)
        # t_kymo = 0:0.1:last_jump+0.5  # +0.5 to show the final state a bit longer
        t_kymo = 0:0.1:20
        num_cells = 3
        num_t = length(t_kymo)
        kymo_data = zeros(Int, num_cells, num_t)
        
        # For each cell, fill in the state at each timepoint (reversed order: Cell 1 above Cell 2)
        for cell in 1:num_cells
            for (j, t) in enumerate(t_kymo)
                # Find the last time before t
                idx = findlast(x -> x <= t, traj.times)
                if isnothing(idx)
                    # Reverse the row index: Cell 1 -> row 2, Cell 2 -> row 1
                    kymo_data[num_cells+1-cell, j] = traj.u_dict[cell][1]
                else
                    # Reverse the row index: Cell 1 -> row 2, Cell 2 -> row 1
                    kymo_data[num_cells+1-cell, j] = traj.u_dict[cell][idx]
                end
            end
        end
        
        # Plot as a heatmap (true kymograph) with Cell 1 above Cell 2
        p_kymo = heatmap(
            t_kymo, 1:num_cells, kymo_data,
            xlabel="Time (arbitrary units)",
            yticks=([cell_num for cell_num in 1:num_cells], ["Cell $(num_cells+1-cell_num)" for cell_num in 1:num_cells]),  # Reversed labels
            color=:matter,
            colorbar_title="State",
            size=(1400, 200),
            legend=false,
            framestyle=:box,
            titlefontsize=22,
            guidefontsize=16,
            tickfontsize=14,
            left_margin=5mm,
            right_margin=5mm,
            top_margin=5mm,
            bottom_margin=5mm,
            grid=false,
            clims=(kymo_min_state, kymo_max_state),  # Use consistent color scale
            colorbar=false  # Disable individual colorbars
        )

        # FIXED: Only show signaling lines when signaling is ON (adjusted for reversed cell order)
        for cell in 1:num_cells
            # Get signaling status at each timepoint with proper error handling
            signaling_status = []
            for t in t_kymo
                idx = findlast(x -> x <= t, traj.times)
                if isnothing(idx)
                    # If no time point found, use the first signaling status
                    push!(signaling_status, traj.s_dict[cell][1])
                else
                    push!(signaling_status, traj.s_dict[cell][idx])
                end
            end
            println("t_kymo: ", t_kymo)
            println("Cell $cell signaling_status: ", signaling_status)
            
            # Find segments where signaling is ON (status == 1) - ONLY PLOT THESE
            i = 1
            while i <= length(signaling_status)
                # Skip segments where signaling is OFF
                while i <= length(signaling_status) && signaling_status[i] != 1
                    i += 1
                end
                
                # If we found an ON segment
                if i <= length(signaling_status)
                    seg_start_idx = i
                    
                    # Find the end of the ON segment
                    while i <= length(signaling_status) && signaling_status[i] == 1
                        i += 1
                    end
                    seg_end_idx = i - 1
                    
                    # Plot only the ON segment (adjusted y position for reversed order)
                    t_start = t_kymo[seg_start_idx]
                    t_end = t_kymo[seg_end_idx]
                    y_val = num_cells+1 - cell  # Reverse the y position: Cell 1 -> y=2, Cell 2 -> y=1
                    
                    plot!(
                        p_kymo, 
                        [t_start, t_end], 
                        [y_val, y_val], 
                        color=:black, 
                        linewidth=2.0,
                        linestyle=:solid, 
                        alpha=1.0,
                        label=false,
                        grid=false,
                    )
                    
                    println("Cell $cell: ON segment from $t_start to $t_end,seg_start_idx: $seg_start_idx, seg_end_idx: $seg_end_idx")
                end
            end
        end
        push!(p_all, p_kymo)
    end
end
# Even more customization: Remove x-axis labels from all but the bottom plot
if length(p_all) > 0
    # Modify plots to remove x-axis labels from all but the last
    for (i, p) in enumerate(p_all)
        if i < length(p_all)
            plot!(p, xlabel="", xticks=false, grid=false)  # Remove x-axis labels for all but last
        else
            plot!(p, xlabel="Time (arbitrary units)", grid=false, xgrid=false, minorgrid=false)  # Keep x-axis label only for bottom plot
        end
        plot!(p)#, title="Trajectory $i")
    end
    
    p_combined_clean = plot(
        p_all..., 
        layout=(length(p_all), 1),
        link=:x,
        size=(1000, 110 * length(p_all)),
        left_margin=5mm,
        right_margin=5mm,
        top_margin=2mm,
        bottom_margin=5mm,
        colorbar=false,
        colorbar_title="State",
        clims=(kymo_min_state, kymo_max_state)  # Apply shared color scale to combined plot
    )
    
    display(p_combined_clean)
    savefig(p_combined_clean, generate_filename(folder_name_for_plots, "multiple_kymographs_stacked_clean_reweighted_cell_3_wins.png"))
    savefig(p_combined_clean, generate_filename(folder_name_for_plots, "multiple_kymographs_stacked_clean_reweighted_cell_3_wins.svg"))
end
