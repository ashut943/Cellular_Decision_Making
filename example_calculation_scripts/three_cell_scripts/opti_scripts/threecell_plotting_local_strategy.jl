# Import required packages
using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO, LaTeXStrings
using Measures
using Revise
using CellularDecisions
using ExponentialUtilities

include("../../../mult_cell/mult_cell.jl")
include("../../../utils/utils.jl")

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
N = 6
M_cell = 3 
K = 3 
h_error = 0.02 
initial_state_array = ((1,0),(1,0),(1,0))  
type_of_boundary_condition = "boundary_2"  
#--------------------------------

#--------------------------------
# Output directories
#--------------------------------
error_str = replace(string(round(h_error*100, digits=1)), "." => "_")
base_folder = joinpath(dirname(dirname(dirname(@__DIR__))), "results", "three_cell_results", "Interior_point_method_results_"*type_of_boundary_condition)
folder_name = joinpath(base_folder, @sprintf("Interior_Point_Method_results_N_%d_error_fix_%s", N, error_str))
mkpath(folder_name)

folder_name_for_plots = joinpath(dirname(dirname(dirname(@__DIR__))), "plots", "three_cell_results", type_of_boundary_condition, "N_$(N)_error_fix_$(error_str)_local_strategy")
mkpath(folder_name_for_plots)
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
initial_state=statedictinv[initial_state_array]+1

threecell_system_filename = generate_filename(folder_name,"threecell_system_local")
threecell_system_local = CellularDecisions.load(threecell_system_filename)

P1_opt_dict = threecell_system_local.parameters_dict[1]
P2_opt_dict = threecell_system_local.parameters_dict[2]
P3_opt_dict = threecell_system_local.parameters_dict[3]
P1_opt_vec_local=CellularDecisions.parameters_to_parameter_vector(P1_opt_dict)
P2_opt_vec_local=CellularDecisions.parameters_to_parameter_vector(P2_opt_dict)
P3_opt_vec_local=CellularDecisions.parameters_to_parameter_vector(P3_opt_dict)
P_opt_dict_local=Dict(1=>P1_opt_vec_local, 2=>P2_opt_vec_local, 3=>P3_opt_vec_local)
Q_opt = Q_maker(P_opt_dict_local, N, M_cell, statedict, statedictinv) 


threecell_system = CellularDecisions.build_cell_system(N, M_cell, Q_opt, threecell_system_local.parameters_dict, type_of_boundary_condition)
threecell_system_filename = generate_filename(folder_name,"threecell_system_local")
CellularDecisions.save(threecell_system,threecell_system_filename)

#--------------------------------
# Display optimal values
#--------------------------------
hitting_times = hitting_time_mod(Q_opt, targetstates_good, targetstates_bad, startstates, 0.0)
hitting_probs = hitting_prob_mod(Q_opt, targetstates_good, targetstates_bad, startstates, 0.0)

println("Expected hitting time: ", hitting_times[initial_state]) 
println("Expected hitting prob: ", hitting_probs[initial_state]) 
println("h_error: ", h_error)
println("------------------------------------------------")
println("F_1^+: ", P1_opt_dict.fp)
println("F_1^-: ", P1_opt_dict.fn)
println("F_2^+: ", P2_opt_dict.fp)
println("F_2^-: ", P2_opt_dict.fn)
println("F_3^+: ", P3_opt_dict.fp)
println("F_3^-: ", P3_opt_dict.fn)
println("G_1: ", P1_opt_dict.gp)
println("G_2: ", P2_opt_dict.gp)
println("G_3: ", P3_opt_dict.gp)
println("k_off_1: ", P1_opt_dict.koff)
println("k_off_2: ", P2_opt_dict.koff)
println("k_off_3: ", P3_opt_dict.koff)
println("------------------------------------------------")

#--------------------------------
# Simulate trajectories
#--------------------------------
T_sim = 50.0
n_sims = 200

println("Simulating CTMC...")
trajectories = [CellularDecisions.simulate_ctmc(Q_opt, initial_state, T_sim, N, M_cell, type_of_boundary_condition) for _ = 1:n_sims]
println("Done simulating CTMC")
targetstates_good_1 = targetstates_good[[statedict[t-1][2][1] ==0 && statedict[t-1][3][1] ==0 for t in targetstates_good]]
targetstates_good_2 = targetstates_good[[statedict[t-1][1][1] ==0 && statedict[t-1][3][1] ==0 for t in targetstates_good]]
targetstates_good_3 = targetstates_good[[statedict[t-1][1][1] ==0 && statedict[t-1][2][1] ==0 for t in targetstates_good]]

terminal_classes = [CellularDecisions.terminal_class(path, [targetstates_good_1, targetstates_good_2, targetstates_good_3], targetstates_bad) for path in trajectories]

failed_trajectories = unpack.(trajectories[terminal_classes .== -1])
success_trajectories = unpack.(trajectories[terminal_classes .== 1 .|| terminal_classes .== 2 .|| terminal_classes .== 3])
success_trajectories_1 = unpack.(trajectories[terminal_classes .== 1])
success_trajectories_2 = unpack.(trajectories[terminal_classes .== 2])
success_trajectories_3 = unpack.(trajectories[terminal_classes .== 3])

total = length(trajectories)
prop_failed = length(failed_trajectories) / total
prop_succ1 = length(success_trajectories_1) / total
prop_succ2 = length(success_trajectories_2) / total 
prop_succ3 = length(success_trajectories_3) / total
# Select representative trajectories based on proportions
num_failed = round(Int, 3 * prop_failed)
num_succ1 = round(Int, 3 * prop_succ1)
num_succ2 = round(Int, 3 * prop_succ2)
num_succ3 = max(round(Int, 3 * prop_succ3), 1)

total_selected = num_failed + num_succ1 + num_succ2 + num_succ3
if total_selected < 3
    num_failed += 3 - total_selected
end

if total_selected > 3
    num_failed -= total_selected - 3
end

representative_trajectories = vcat(
    length(failed_trajectories) > 0 ? rand(failed_trajectories, min(num_failed, length(failed_trajectories))) : [],
    length(success_trajectories_1) > 0 ? rand(success_trajectories_1, min(num_succ1, length(success_trajectories_1))) : [],
    length(success_trajectories_2) > 0 ? rand(success_trajectories_2, min(num_succ2, length(success_trajectories_2))) : [],
    length(success_trajectories_3) > 0 ? rand(success_trajectories_3, min(num_succ3, length(success_trajectories_3))) : []
)

println("Success trajectories 1: ", length(success_trajectories_1))
println("Success trajectories 2: ", length(success_trajectories_2))
println("Success trajectories 3: ", length(success_trajectories_3))
println("Success trajectories: ", length(success_trajectories))
println("Failed trajectories: ", length(failed_trajectories))

#--------------------------------
# Plot trajectories
#--------------------------------
trajectories_unpacked = unpack.(trajectories)
p_traj = plot()
for i = 1:min(length(trajectories_unpacked), 50)
    curr_state = trajectories_unpacked[i].u_dict
    plot_ctmc!(p_traj, trajectories_unpacked[i].times, curr_state[1], trajectories_unpacked[i].final_time, c=:blue, linewidth=0.2)
    plot_ctmc!(p_traj, trajectories_unpacked[i].times, curr_state[2], trajectories_unpacked[i].final_time, c=:red, linewidth=0.2)
    plot_ctmc!(p_traj, trajectories_unpacked[i].times, curr_state[3], trajectories_unpacked[i].final_time, c=:green, linewidth=0.2)
end
display(p_traj)
savefig(p_traj, generate_filename(folder_name_for_plots, "multiple_trajectories_local.png"))
savefig(p_traj, generate_filename(folder_name_for_plots, "multiple_trajectories_local.svg"))

t_plot = 0:0.3:50
u1_avg_fail, u2_avg_fail, u3_avg_fail, u1_std_fail, u2_std_fail, u3_std_fail = average_trajectory(failed_trajectories, t_plot)
u1_avg_succ, u2_avg_succ, u3_avg_succ, u1_std_succ, u2_std_succ, u3_std_succ = average_trajectory(success_trajectories_3, t_plot)

p_fail = plot(t_plot, u1_avg_fail, ribbon=u1_std_fail, label="cell 1", xlabel="Time", ylabel="State", title="Unsuccessful trajectories", color=:purple)
plot!(p_fail, t_plot, u2_avg_fail, ribbon=u2_std_fail, label="cell 2", color=:green)
plot!(p_fail, t_plot, u3_avg_fail, ribbon=u3_std_fail, label="cell 3", color=:brown)
savefig(p_fail, generate_filename(folder_name_for_plots, "multiple_trajectories_local_avg_fail.png")) 
savefig(p_fail, generate_filename(folder_name_for_plots, "multiple_trajectories_local_avg_fail.svg"))

p_succ = plot(t_plot, u1_avg_succ, ribbon=u1_std_succ, label="cell 1", xlabel="Time", ylabel="State", title="Successful trajectories", color=:purple)
plot!(p_succ, t_plot, u2_avg_succ, ribbon=u2_std_succ, label="cell 2", color=:green)
plot!(p_succ, t_plot, u3_avg_succ, ribbon=u3_std_succ, label="cell 3", color=:brown)
savefig(p_succ, generate_filename(folder_name_for_plots, "multiple_trajectories_local_avg_success.png"))
savefig(p_succ, generate_filename(folder_name_for_plots, "multiple_trajectories_local_avg_success.svg"))

p_both = plot(p_fail, p_succ, layout=(2,1))
display(p_both)
savefig(p_both, generate_filename(folder_name_for_plots, "multiple_trajectories_local_avg_both_success_and_fail.png"))
savefig(p_both, generate_filename(folder_name_for_plots, "multiple_trajectories_local_avg_both_success_and_fail.svg"))


#--------------------------------
# Kymograph plots
#--------------------------------
kymograph_plots = []
num_kymographs = 3

global kymo_min_state = Inf
global kymo_max_state = -Inf

if length(representative_trajectories) > 0
    for idx in 1:min(num_kymographs, length(representative_trajectories))
        traj = representative_trajectories[idx]
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
    error("No valid trajectory data for kymograph")
end

if length(representative_trajectories) > 0
    println("Kymograph color scale: $kymo_min_state to $kymo_max_state")
    
    for idx in 1:num_kymographs
        traj = representative_trajectories[idx]
        t_kymo = 0:0.1:20
        num_cells = 3
        num_t = length(t_kymo)
        kymo_data = zeros(Int, num_cells, num_t)
        
        for cell in 1:num_cells
            for (j, t) in enumerate(t_kymo)
                time_idx = findlast(x -> x <= t, traj.times)
                if isnothing(time_idx)
                    kymo_data[num_cells+1-cell, j] = traj.u_dict[cell][1]
                else
                    kymo_data[num_cells+1-cell, j] = traj.u_dict[cell][time_idx]
                end
            end
        end
        
        p_kymo = heatmap(
            t_kymo, 1:num_cells, kymo_data,
            xlabel="Time (arbitrary units)",
            yticks=([cell_num for cell_num in 1:num_cells], ["Cell $(num_cells+1-cell_num)" for cell_num in 1:num_cells]),  # Reversed labels
            color=:YlOrRd_7,
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
            clims=(kymo_min_state, kymo_max_state),
            colorbar=false
        )

        # Add signaling ON segments as black lines
        for cell in 1:num_cells
            signaling_status = Int[]
            for t in t_kymo
                time_idx = findlast(x -> x <= t, traj.times)
                if isnothing(time_idx)
                    push!(signaling_status, traj.s_dict[cell][1])
                else
                    push!(signaling_status, traj.s_dict[cell][time_idx])
                end
            end
            
            i = 1
            while i <= length(signaling_status)
                while i <= length(signaling_status) && signaling_status[i] != 1
                    i += 1
                end
                
                if i <= length(signaling_status)
                    seg_start = i
                    while i <= length(signaling_status) && signaling_status[i] == 1
                        i += 1
                    end
                    seg_end = i - 1
                    
                    t_start = t_kymo[seg_start]
                    t_end = t_kymo[seg_end]
                    y_val = num_cells + 1 - cell
                    
                    plot!(p_kymo, [t_start, t_end], [y_val, y_val], 
                        color=:black, linewidth=2.0, linestyle=:solid, alpha=1.0, label=false, grid=false)
                end
            end
        end
        push!(kymograph_plots, p_kymo)
    end
end

if length(kymograph_plots) > 0
    for (i, p) in enumerate(kymograph_plots)
        if i < length(kymograph_plots)
            plot!(p, xlabel="", xticks=false, grid=false)
        else
            plot!(p, xlabel="Time (arbitrary units)", grid=false, xgrid=false, minorgrid=false)
        end
        plot!(p)
    end
    
    p_kymographs_combined = plot(
        kymograph_plots..., 
        layout=(length(kymograph_plots), 1),
        link=:x,
        size=(1000, 110 * length(kymograph_plots)),
        left_margin=5mm,
        right_margin=5mm,
        top_margin=2mm,
        bottom_margin=5mm,
        colorbar=false,
        colorbar_title="State",
        clims=(kymo_min_state, kymo_max_state)
    )
    
    display(p_kymographs_combined)
    savefig(p_kymographs_combined, generate_filename(folder_name_for_plots, "multiple_kymographs_stacked_clean_local.png"))
    savefig(p_kymographs_combined, generate_filename(folder_name_for_plots, "multiple_kymographs_stacked_clean_local.svg"))
end

#--------------------------------
# Parameter heatmaps
#--------------------------------
f_plus_data = zeros(Int(length(P1_opt_dict.fp)/2), 2)
f_plus_data[:, 1] = P1_opt_dict.fp[1:N]
f_plus_data[:, 2] = P1_opt_dict.fp[N+1:2*N]

f_minus_data = zeros(Int(length(P1_opt_dict.fn)/2), 2)
f_minus_data[:, 1] = P1_opt_dict.fn[1:N]
f_minus_data[:, 2] = P1_opt_dict.fn[N+1:2*N]

g_data = P1_opt_dict.gp
if isnothing(g_data) || isempty(g_data)
    error("g_data is nothing or empty")
end
g_data = reshape(g_data, :, 1)

k_off_data = P1_opt_dict.koff
if isnothing(k_off_data) || isempty(k_off_data)
    error("k_off_data is nothing or empty")
end
k_off_data = reshape([k_off_data], 1, 1)

heatmap_clim = (
    min(
        minimum(f_plus_data),
        minimum(f_minus_data), 
        minimum(g_data),
        minimum(k_off_data)
    ),
    max(
        maximum(f_plus_data),
        maximum(f_minus_data),
        maximum(g_data), 
        maximum(k_off_data)
    )
)

p_f_plus = heatmap(f_plus_data,
    xticks=([1, 2], ["0", "1"]),
    yticks=(1:N, 0:N-1),
    color=:acton,
    size=(100, 50*N),
    titlefontsize=14, guidefontsize=12, tickfontsize=10,
    grid=true, gridstyle=:dash, gridalpha=1.0,
    framestyle=:box, clims=heatmap_clim, colorbar=true)

p_f_minus = heatmap(f_minus_data,
    xticks=([1, 2], ["0", "1"]),
    yticks=1:N,
    color=:acton,
    size=(100, 50*N),
    titlefontsize=14, guidefontsize=12, tickfontsize=10,
    grid=true, gridstyle=:dash, gridalpha=1.0,
    framestyle=:box, clims=heatmap_clim, colorbar=true)

display(plot(p_f_plus, p_f_minus, layout=(1,2), size=(300,300)))

p_g = heatmap(g_data,
    yticks=(1:N+1, 0:N),
    xticks=false,
    color=:acton,
    size=(50, 50*(N+1)),
    titlefontsize=14, guidefontsize=12, tickfontsize=10,
    framestyle=:box, clims=heatmap_clim, colorbar=true)
for y in 1:N+1
    hline!(p_g, [y-0.3], color=:black, lw=0.3, label=nothing)
end
display(p_g)

p_k_off = heatmap(k_off_data',
    xticks=false, yticks=false,
    color=:acton,
    size=(50, 50),
    titlefontsize=14, guidefontsize=12, tickfontsize=10,
    framestyle=:box, clims=heatmap_clim, colorbar=true)
display(p_k_off)

savefig(p_f_plus, generate_filename(folder_name_for_plots, "f_plus_heatmap_with_colorbar.png"))
savefig(p_f_plus, generate_filename(folder_name_for_plots, "f_plus_heatmap_with_colorbar.svg"))

savefig(p_f_minus, generate_filename(folder_name_for_plots, "f_minus_heatmap_with_colorbar.png"))
savefig(p_f_minus, generate_filename(folder_name_for_plots, "f_minus_heatmap_with_colorbar.svg"))

savefig(p_g, generate_filename(folder_name_for_plots, "g_heatmap_with_colorbar.png"))
savefig(p_g, generate_filename(folder_name_for_plots, "g_heatmap_with_colorbar.svg"))

savefig(p_k_off, generate_filename(folder_name_for_plots, "k_off_heatmap_with_colorbar.png"))
savefig(p_k_off, generate_filename(folder_name_for_plots, "k_off_heatmap_with_colorbar.svg"))