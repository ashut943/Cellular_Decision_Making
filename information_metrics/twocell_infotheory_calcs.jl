# Import required packages
using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO
using Revise
using Plots.PlotMeasures
using CellularDecisions_final
using NumericalIntegration
using Interpolations
using Distributed
using SharedArrays



# Include helper files
include("../mult_cell/mult_cell_setup.jl")
include("../mult_cell/mult_cell_nonlinear.jl")
include("../mult_cell/mult_cell_hittingtime.jl")
include("../utils/ctmc_vis.jl")
include("../utils/ctmc_core.jl")
include("../utils/file_utils.jl")
include("../utils/video_utils.jl")
include("../information_metrics/infotheoryfuncs_two_cells.jl")
include("../two_cell_experiments/two_cell_coarse_graining.jl")

#--------------------------------
#Calculate information metrics for single trajectory

function calculate_information_metrics_single_trajectory(cell_system, trajectory, ts_full, starting_state_tuple)

    local starting_state_tuple_coarse=coarse_grain_tuple(starting_state_tuple)
    local x_initial=starting_state_tuple_coarse[1]
    local y_initial=starting_state_tuple_coarse[2]
    local z_initial=starting_state_tuple_coarse[3]

    local starting_state_index_XY=hidden_var_to_index(z_initial)#2*s_a_initial+s_b_initial+1
    local starting_state_index_X=hidden_var_y_to_index(y_initial,z_initial)#4*u_b_initial+2*s_a_initial+s_b_initial+s_c_initial+1
    local starting_state_index_Y=hidden_var_x_to_index(x_initial,z_initial)#4*u_a_initial+2*s_a_initial+s_b_initial+s_c_initial+1

    local x_neighbours, y_neighbours, x_boundary, y_boundary = get_neighbours(cell_system)
    local x_coarse_temp, y_coarse_temp, t_coarse_temp, _, _, _ = coarse_grained_paths(trajectory)
    local x_coarse_x_temp, y_coarse_x_temp, t_coarse_x_temp = coarse_grained_paths_x(x_coarse_temp, y_coarse_temp, t_coarse_temp)
    local x_coarse_y_temp, y_coarse_y_temp, t_coarse_y_temp = coarse_grained_paths_y(x_coarse_temp, y_coarse_temp, t_coarse_temp)
    
    local ts_temp = unique(sort(vcat(ts_full, t_coarse_temp)))
    ts_temp = Float64.(ts_temp)  # Convert to Float64
    local t_coarse_x_index_in_t_coarse = Int[]
    for tx in t_coarse_x_temp
        indices = findall(t -> t == tx, t_coarse_temp)
        append!(t_coarse_x_index_in_t_coarse, indices)
    end
    local t_coarse_y_index_in_t_coarse = Int[]
    for ty in t_coarse_y_temp
        indices = findall(t -> t == ty, t_coarse_temp)
        append!(t_coarse_y_index_in_t_coarse, indices)
    end
    #--------------------------------
    #Classifying jump times
    #first getting jump times in t_coarse_temp
    local x_jump_times_index=t_coarse_x_index_in_t_coarse[2:end]
    local y_jump_times_index=t_coarse_y_index_in_t_coarse[2:end]
    local x_at_jump_times=[]
    for i in 1:length(x_jump_times_index)
        push!(x_at_jump_times, x_coarse_temp[x_jump_times_index[i]-1])
    end

    #getting states at and after jump times
    local x_after_jump_times=x_coarse_temp[x_jump_times_index]
    local y_at_jump_times=[]
    for i in 1:length(y_jump_times_index)
        push!(y_at_jump_times, y_coarse_temp[y_jump_times_index[i]-1])
    end
    local y_after_jump_times=y_coarse_temp[y_jump_times_index]

    #Now classifying jump times in ts_temp based on positive or negative jumps
    local t_jumps_x_neighbours_index_in_t_coarse=Dict{Any, Vector{Int64}}()
    local t_jumps_y_neighbours_index_in_t_coarse=Dict{Any, Vector{Int64}}()
    for x_neighbour in x_neighbours
        t_jumps_x_neighbours_index_in_t_coarse[x_neighbour]=[]
    end
    for y_neighbour in y_neighbours
        t_jumps_y_neighbours_index_in_t_coarse[y_neighbour]=[]
    end

    for i in 1:length(x_jump_times_index)
        for x_neighbour_increment in x_neighbours
            if x_at_jump_times[i].+x_neighbour_increment == x_after_jump_times[i]
                push!(t_jumps_x_neighbours_index_in_t_coarse[x_neighbour_increment], x_jump_times_index[i])
            end
        end
    end
    for i in 1:length(y_jump_times_index)
        for y_neighbour_increment in y_neighbours
            if y_at_jump_times[i].+y_neighbour_increment == y_after_jump_times[i]
                push!(t_jumps_y_neighbours_index_in_t_coarse[y_neighbour_increment], y_jump_times_index[i])
            end
        end
    end

    #--------------------------------
    #Now getting jump indices in ts
    local t_jumps_x_neighbours_index = Dict{Any, Vector{Int64}}()
    for (x_neighbour_index,x_neighbour_increment) in enumerate(x_neighbours)
        t_jumps_x_neighbours_index[x_neighbour_increment]=findall(t -> t ∈ t_coarse_temp[t_jumps_x_neighbours_index_in_t_coarse[x_neighbour_increment]], ts_temp)
    end
    local t_jumps_y_neighbours_index = Dict{Any, Vector{Int64}}()
    for (y_neighbour_index,y_neighbour_increment) in enumerate(y_neighbours)
        t_jumps_y_neighbours_index[y_neighbour_increment]=findall(t -> t ∈ t_coarse_temp[t_jumps_y_neighbours_index_in_t_coarse[y_neighbour_increment]], ts_temp)
    end
    #--------------------------------
    local W_XY_X_neighbours=Dict{Any, Vector{Float64}}()
    local W_XY_Y_neighbours=Dict{Any, Vector{Float64}}() 
    for (x_neighbour_index,x_neighbour_increment) in enumerate(x_neighbours)
        W_XY_X_neighbours[x_neighbour_increment]=zeros(length(ts_temp))
    end
    for (y_neighbour_index,y_neighbour_increment) in enumerate(y_neighbours)
        W_XY_Y_neighbours[y_neighbour_increment]=zeros(length(ts_temp))
    end
    
    for i in 1:length(ts_temp)
        t=ts_temp[i]
        W_XY_X_curr, W_XY_Y_curr = W_XY_trajectory(t, cell_system, trajectory, starting_state_index_XY)
        for (x_neighbour_index,x_neighbour_increment) in enumerate(x_neighbours)
            W_XY_X_neighbours[x_neighbour_increment][i]=W_XY_X_curr[x_neighbour_increment]
        end
        for (y_neighbour_index,y_neighbour_increment) in enumerate(y_neighbours)
            W_XY_Y_neighbours[y_neighbour_increment][i]=W_XY_Y_curr[y_neighbour_increment]
        end
    end

    
    local W_X_X_neighbours=Dict{Any, Vector{Float64}}()
    local W_Y_Y_neighbours=Dict{Any, Vector{Float64}}()
    for (x_neighbour_index,x_neighbour_increment) in enumerate(x_neighbours)
        W_X_X_neighbours[x_neighbour_increment]=zeros(length(ts_temp))
    end
    for (y_neighbour_index,y_neighbour_increment) in enumerate(y_neighbours)
        W_Y_Y_neighbours[y_neighbour_increment]=zeros(length(ts_temp))
    end
    
    for i in 1:length(ts_temp)
        t = ts_temp[i]
        W_X_X_curr = W_X_trajectory(t, cell_system, trajectory, starting_state_index_X)
        for (x_neighbour_index,x_neighbour_increment) in enumerate(x_neighbours)
            W_X_X_neighbours[x_neighbour_increment][i]=W_X_X_curr[x_neighbour_increment]
        end
    end
    for i in 1:length(ts_temp)
        t = ts_temp[i]
        W_Y_Y_curr = W_Y_trajectory(t, cell_system, trajectory, starting_state_index_Y)
        for (y_neighbour_index,y_neighbour_increment) in enumerate(y_neighbours)
            W_Y_Y_neighbours[y_neighbour_increment][i]=W_Y_Y_curr[y_neighbour_increment]
        end
    end
    #--------------------------------
    local integrand_X_neighbours_1=Dict{Any, Vector{Float64}}()#zeros(Float64,length(x_neighbours),length(ts_temp))
    local integrand_X_neighbours_2=Dict{Any, Vector{Float64}}()#zeros(Float64,length(x_neighbours),length(ts_temp))
    local integrand_Y_neighbours_1=Dict{Any, Vector{Float64}}()#zeros(Float64,length(y_neighbours),length(ts_temp))
    local integrand_Y_neighbours_2=Dict{Any, Vector{Float64}}()#zeros(Float64,length(y_neighbours),length(ts_temp))

    for (x_neighbour_index, x_neighbour_increment) in enumerate(x_neighbours)
        integrand_X_neighbours_1[x_neighbour_increment]=zeros(length(ts_temp))
        integrand_X_neighbours_2[x_neighbour_increment]=zeros(length(ts_temp))
    end
    for (y_neighbour_index, y_neighbour_increment) in enumerate(y_neighbours)
        integrand_Y_neighbours_1[y_neighbour_increment]=zeros(length(ts_temp))
        integrand_Y_neighbours_2[y_neighbour_increment]=zeros(length(ts_temp))
    end
    for i in 1:length(ts_temp)
        # println("i: ", i, "length(ts_temp): ", length(ts_temp))
        for (x_neighbour_index, x_neighbour_increment) in enumerate(x_neighbours)
            curr_W_XY_X_neighbours=W_XY_X_neighbours[x_neighbour_increment][i]
            curr_W_X_X_neighbours=W_X_X_neighbours[x_neighbour_increment][i]
            integrand_X_neighbours_2[x_neighbour_increment][i] = curr_W_XY_X_neighbours-curr_W_X_X_neighbours
            if curr_W_XY_X_neighbours != 0 && curr_W_X_X_neighbours != 0
                integrand_X_neighbours_1[x_neighbour_increment][i] = (log.(curr_W_XY_X_neighbours) - log.(curr_W_X_X_neighbours)) * curr_W_XY_X_neighbours
            end
            # if x_neighbour_increment == -1 && integrand_Xn_1[i]!=0
            #     println(integrand_X_neighbours_1[x_neighbour_increment][i]," ",integrand_Xn_1[i])
            #     println(integrand_X_neighbours_2[x_neighbour_increment][i]," ",integrand_Xn_2[i])
            #     println(integrand_X_neighbours_1[x_neighbour_increment][i]-integrand_X_neighbours_2[x_neighbour_increment][i])
            #     println(integrand_Xn_1[i]-integrand_Xn_2[i])
            # end
        end
        for (y_neighbour_index, y_neighbour_increment) in enumerate(y_neighbours)
            curr_W_XY_Y_neighbours=W_XY_Y_neighbours[y_neighbour_increment][i]
            curr_W_Y_Y_neighbours=W_Y_Y_neighbours[y_neighbour_increment][i]
            integrand_Y_neighbours_2[y_neighbour_increment][i] = curr_W_XY_Y_neighbours-curr_W_Y_Y_neighbours
            if curr_W_XY_Y_neighbours > 0 && curr_W_Y_Y_neighbours > 0
                integrand_Y_neighbours_1[y_neighbour_increment][i] = (log.(curr_W_XY_Y_neighbours) - log.(curr_W_Y_Y_neighbours)) * curr_W_XY_Y_neighbours
            end
        end
    end


    local integrand_neighbourwise_XY=Dict{Any, Vector{Float64}}()#zeros(Float64, length(x_neighbours), length(ts_temp))
    local integrand_neighbourwise_YX=Dict{Any, Vector{Float64}}()#zeros(Float64, length(y_neighbours), length(ts_temp))
    for (x_neighbour_index, x_neighbour_increment) in enumerate(x_neighbours)
        integrand_neighbourwise_XY[x_neighbour_increment]=integrand_X_neighbours_1[x_neighbour_increment] - integrand_X_neighbours_2[x_neighbour_increment]
    end
    for (y_neighbour_index, y_neighbour_increment) in enumerate(y_neighbours)
        integrand_neighbourwise_YX[y_neighbour_increment]=integrand_Y_neighbours_1[y_neighbour_increment] - integrand_Y_neighbours_2[y_neighbour_increment]
    end

    #calculating the continuous parts of te y->x
    local curr_integrand_transfer_entropy_XY=zeros(Float64, length(ts_temp))
    local curr_integrand_transfer_entropy_YX=zeros(Float64, length(ts_temp))
    for i in 1:length(ts_temp)
        curr_integrand_transfer_entropy_XY[i] = sum(integrand_neighbourwise_XY[x_neighbour_increment][i] for x_neighbour_increment in x_neighbours)
        curr_integrand_transfer_entropy_YX[i] = sum(integrand_neighbourwise_YX[y_neighbour_increment][i] for y_neighbour_increment in y_neighbours)
    end
    #--------------------------------
    #calculate the discontinuous parts of te y->x
    local relevant_W_x_xneighbours=Dict{Any, Vector{Float64}}()
    local log_relevant_W_x_xneighbours=Dict{Any, Vector{Float64}}()
    local relevant_W_xy_xneighbours=Dict{Any, Vector{Float64}}()
    local log_relevant_W_xy_xneighbours=Dict{Any, Vector{Float64}}()
    local relevant_W_y_yneighbours=Dict{Any, Vector{Float64}}()
    local log_relevant_W_y_yneighbours=Dict{Any, Vector{Float64}}()
    local relevant_W_xy_yneighbours=Dict{Any, Vector{Float64}}()
    local log_relevant_W_xy_yneighbours=Dict{Any, Vector{Float64}}()
    local log_relevant_diff_x_xneighbours_cumsum=Dict{Any, Vector{Float64}}()
    local log_relevant_diff_y_yneighbours_cumsum=Dict{Any, Vector{Float64}}()
    for (x_neighbour_index, x_neighbour_increment) in enumerate(x_neighbours)
        t_jumps_relevant_x_xneighbours=t_jumps_x_neighbours_index[x_neighbour_increment]
        relevant_W_x_xneighbours[x_neighbour_increment]=W_X_X_neighbours[x_neighbour_increment][t_jumps_relevant_x_xneighbours]
        relevant_W_xy_xneighbours[x_neighbour_increment]=W_XY_X_neighbours[x_neighbour_increment][t_jumps_relevant_x_xneighbours]
        log_relevant_W_x_xneighbours[x_neighbour_increment]=log.(relevant_W_x_xneighbours[x_neighbour_increment])
        log_relevant_W_xy_xneighbours[x_neighbour_increment]=log.(relevant_W_xy_xneighbours[x_neighbour_increment])
    end
    for (y_neighbour_index, y_neighbour_increment) in enumerate(y_neighbours)
        t_jumps_relevant_y_yneighbours=t_jumps_y_neighbours_index[y_neighbour_increment]
        relevant_W_y_yneighbours[y_neighbour_increment]=W_Y_Y_neighbours[y_neighbour_increment][t_jumps_relevant_y_yneighbours]
        relevant_W_xy_yneighbours[y_neighbour_increment]=W_XY_Y_neighbours[y_neighbour_increment][t_jumps_relevant_y_yneighbours]
        log_relevant_W_y_yneighbours[y_neighbour_increment]=log.(relevant_W_y_yneighbours[y_neighbour_increment])
        log_relevant_W_xy_yneighbours[y_neighbour_increment]=log.(relevant_W_xy_yneighbours[y_neighbour_increment])
    end

    local log_relevant_diff_x_xneighbours_temp=Dict{Any, Vector{Float64}}()
    local log_relevant_diff_y_yneighbours_temp=Dict{Any, Vector{Float64}}()
    local log_relevant_diff_x_xneighbours_tots=Dict{Any, Vector{Float64}}()
    local log_relevant_diff_y_yneighbours_tots=Dict{Any, Vector{Float64}}()
    for (x_neighbour_index,x_neighbour_increment) in enumerate(x_neighbours)
        local curr_log_relevant_W_xy_xneighbours=log_relevant_W_xy_xneighbours[x_neighbour_increment]
        local curr_log_relevant_W_x_xneighbours=log_relevant_W_x_xneighbours[x_neighbour_increment]
        local curr_relevant_W_x_xneighbours=relevant_W_x_xneighbours[x_neighbour_increment]
        local curr_relevant_W_xy_xneighbours=relevant_W_xy_xneighbours[x_neighbour_increment]
        local log_relevant_diff_x_xneighbours_temp[x_neighbour_increment]=[]
        local t_jumps_relevant_x_xneighbours_index=t_jumps_x_neighbours_index[x_neighbour_increment]
        if !isempty(curr_log_relevant_W_x_xneighbours)
            for i in 1:length(curr_log_relevant_W_x_xneighbours)
                if curr_relevant_W_xy_xneighbours[i] > 0
                    if curr_relevant_W_x_xneighbours[i] > 0
                        push!(log_relevant_diff_x_xneighbours_temp[x_neighbour_increment], curr_log_relevant_W_xy_xneighbours[i] - curr_log_relevant_W_x_xneighbours[i])
                    else
                        push!(log_relevant_diff_x_xneighbours_temp[x_neighbour_increment], curr_log_relevant_W_xy_xneighbours[i])
                    end
                else
                    if curr_relevant_W_x_xneighbours[i] > 0
                        push!(log_relevant_diff_x_xneighbours_temp[x_neighbour_increment], -curr_log_relevant_W_x_xneighbours[i])
                    else
                        push!(log_relevant_diff_x_xneighbours_temp[x_neighbour_increment], 0)
                    end
                end
            end
        end
        local log_relevant_diff_x_xneighbours=zeros(length(ts_temp))
        log_relevant_diff_x_xneighbours[t_jumps_relevant_x_xneighbours_index]=log_relevant_diff_x_xneighbours_temp[x_neighbour_increment]
        log_relevant_diff_x_xneighbours_tots[x_neighbour_increment]=log_relevant_diff_x_xneighbours
        log_relevant_diff_x_xneighbours_cumsum[x_neighbour_increment]=cumsum(log_relevant_diff_x_xneighbours)
    end

    for (y_neighbour_index,y_neighbour_increment) in enumerate(y_neighbours)
        local curr_log_relevant_W_xy_yneighbours=log_relevant_W_xy_yneighbours[y_neighbour_increment]
        local curr_log_relevant_W_y_yneighbours=log_relevant_W_y_yneighbours[y_neighbour_increment]
        local curr_relevant_W_y_yneighbours=relevant_W_y_yneighbours[y_neighbour_increment]
        local curr_relevant_W_xy_yneighbours=relevant_W_xy_yneighbours[y_neighbour_increment]
        local log_relevant_diff_y_yneighbours_temp[y_neighbour_increment]=[]
        local t_jumps_relevant_y_yneighbours_index=t_jumps_y_neighbours_index[y_neighbour_increment]
        if !isempty(curr_log_relevant_W_y_yneighbours)
            for i in 1:length(curr_log_relevant_W_y_yneighbours)
                if curr_relevant_W_xy_yneighbours[i] > 0
                    if curr_relevant_W_y_yneighbours[i] > 0
                        push!(log_relevant_diff_y_yneighbours_temp[y_neighbour_increment], curr_log_relevant_W_xy_yneighbours[i] - curr_log_relevant_W_y_yneighbours[i])
                    else
                        push!(log_relevant_diff_y_yneighbours_temp[y_neighbour_increment], curr_log_relevant_W_xy_yneighbours[i])
                    end
                else
                    if curr_relevant_W_y_yneighbours[i] > 0
                        push!(log_relevant_diff_y_yneighbours_temp[y_neighbour_increment], -curr_log_relevant_W_y_yneighbours[i])
                    else
                        push!(log_relevant_diff_y_yneighbours_temp[y_neighbour_increment], 0)
                    end
                end
            end
        end
        local log_relevant_diff_y_yneighbours=zeros(length(ts_temp))
        log_relevant_diff_y_yneighbours[t_jumps_relevant_y_yneighbours_index]=log_relevant_diff_y_yneighbours_temp[y_neighbour_increment]
        log_relevant_diff_y_yneighbours_tots[y_neighbour_increment]=log_relevant_diff_y_yneighbours
        log_relevant_diff_y_yneighbours_cumsum[y_neighbour_increment]=cumsum(log_relevant_diff_y_yneighbours)
    end
    local discts_parts_xy=zeros(length(ts_temp))
    local discts_parts_yx=zeros(length(ts_temp))
    local log_relevant_diff_x=zeros(length(ts_temp))
    local log_relevant_diff_y=zeros(length(ts_temp))
    for (x_neighbour_index,x_neighbour_increment) in enumerate(x_neighbours)
        discts_parts_xy+=log_relevant_diff_x_xneighbours_cumsum[x_neighbour_increment]
        log_relevant_diff_x+=log_relevant_diff_x_xneighbours_tots[x_neighbour_increment]
    end
    for (y_neighbour_index,y_neighbour_increment) in enumerate(y_neighbours)
        discts_parts_yx+=log_relevant_diff_y_yneighbours_cumsum[y_neighbour_increment]
        log_relevant_diff_y+=log_relevant_diff_y_yneighbours_tots[y_neighbour_increment]
    end
    #--------------------------------
    #calculating the relevant integrals
    local integral_X_xneighbours_1=Dict{Any, Vector{Float64}}()
    local integral_X_xneighbours_2=Dict{Any, Vector{Float64}}()
    local integral_Y_yneighbours_1=Dict{Any, Vector{Float64}}()
    local integral_Y_yneighbours_2=Dict{Any, Vector{Float64}}()
    for (x_neighbour_index,x_neighbour_increment) in enumerate(x_neighbours)
        integral_X_xneighbours_1[x_neighbour_increment]=cumul_integrate(ts_temp, integrand_X_neighbours_1[x_neighbour_increment])
        integral_X_xneighbours_2[x_neighbour_increment]=cumul_integrate(ts_temp, integrand_X_neighbours_2[x_neighbour_increment])
    end
    for (y_neighbour_index,y_neighbour_increment) in enumerate(y_neighbours)
        integral_Y_yneighbours_1[y_neighbour_increment]=cumul_integrate(ts_temp, integrand_Y_neighbours_1[y_neighbour_increment])
        integral_Y_yneighbours_2[y_neighbour_increment]=cumul_integrate(ts_temp, integrand_Y_neighbours_2[y_neighbour_increment])
    end
    #--------------------------------
    #calculating the transfer entropies and mutual information now
    #first Y->X
    local path_te_XY_temp=zeros(length(ts_temp))
    local path_te_XY_rate_for_single_path_temp=zeros(length(ts_temp))
    for (x_neighbour_index,x_neighbour_increment) in enumerate(x_neighbours)
        path_te_XY_temp+=integral_X_xneighbours_2[x_neighbour_increment]
        path_te_XY_rate_for_single_path_temp+=integrand_X_neighbours_2[x_neighbour_increment]
    end
    local path_te_YX_temp=zeros(length(ts_temp))
    local path_te_YX_rate_for_single_path_temp=zeros(length(ts_temp))
    for (y_neighbour_index,y_neighbour_increment) in enumerate(y_neighbours)
        path_te_YX_temp+=integral_Y_yneighbours_2[y_neighbour_increment]
        path_te_YX_rate_for_single_path_temp+=integrand_Y_neighbours_2[y_neighbour_increment]
    end
    local path_te_XY = discts_parts_xy - path_te_XY_temp#integral_Xp_2 - integral_Xn_2
    local path_te_XY_rate = curr_integrand_transfer_entropy_XY #log_relevant_diff_x <- for single path
    local path_te_XY_rate_for_single_path = log_relevant_diff_x - path_te_XY_rate_for_single_path_temp
    #then X->Y
    local path_te_YX = discts_parts_yx - path_te_YX_temp
    local path_te_YX_rate = curr_integrand_transfer_entropy_YX #log_relevant_diff_y
    local path_te_YX_rate_for_single_path = log_relevant_diff_y - path_te_YX_rate_for_single_path_temp
    #now mutual information
    local path_mi = path_te_XY + path_te_YX
    local path_mi_rate = path_te_XY_rate + path_te_YX_rate
    local path_mi_rate_for_single_path = path_te_XY_rate_for_single_path+path_te_YX_rate_for_single_path
    #--------------------------------
    #older calculations:
    # XY_transfer_entropy = integral_Xp_1 + integral_Xn_1 - integral_Xp_2 - integral_Xn_2
    # YX_transfer_entropy = integral_Yp_1 + integral_Yn_1 - integral_Yp_2 - integral_Yn_2
    return ts_temp,path_te_XY, path_te_YX, path_mi, path_te_XY_rate, path_te_YX_rate, path_mi_rate, path_te_XY_rate_for_single_path, path_te_YX_rate_for_single_path, path_mi_rate_for_single_path
end

#--------------------------------
# Calculate information metrics for multiple trajectories
function calculate_information_metrics_multiple_trajectories(twocell_system, starting_state_tuple; T=50.0, initial_state=1, num_simulations=100, num_timesteps=1000)
    
    #get initial state tuple
    # Create plots directory if it doesn't exist    
    N = twocell_system.internal_states
    Q = twocell_system.Q_matrix
    TG = twocell_system.T_good
    TB = twocell_system.T_bad
    Tc = twocell_system.Tc
    statedict=twocell_system.state_dict
    statedictinv=twocell_system.state_dict_inv 

    ni, np = CellularDecisions_final.varioussizes(N,2)
    ns=length(Tc);
    targetstates_good = [target_state+1 for target_state ∈ TG]  # Good target states
    targetstates_bad = [target_state+1 for target_state ∈ TB]   # Bad target states
    targetstates = [targetstates_good; targetstates_bad]        # All target states
    startstates = [start_state+1 for start_state ∈ Tc]         # Starting states
    allstates = [startstates; targetstates_good; targetstates_bad]
        
    # Simulate trajectories
    S_arr = [CellularDecisions_final.simulate_ctmc(Q, initial_state, T, N,2, "boundary_2") for i = 1:num_simulations]
    targetstates_good_a = targetstates_good[[statedict[t-1][2][1] == 0 for t in targetstates_good]]
    targetstates_good_b = setdiff(targetstates_good, targetstates_good_a)
    
    terminal_classes = [CellularDecisions_final.terminal_class(path, [targetstates_good_a, targetstates_good_b], targetstates_bad) for path in S_arr]
    
    failed_trajectories = (S_arr[terminal_classes .== -1])
    success_trajectories = (S_arr[terminal_classes .== 1 .|| terminal_classes .== 2])
    success_trajectories_a = (S_arr[terminal_classes .== 1])
    success_trajectories_b = (S_arr[terminal_classes .== 2])
    println("Success trajectories: ", length(success_trajectories))
    println("Failed trajectories: ", length(failed_trajectories))
    
    # Find maximum coarse-grained time
    find_max_t_coarse = []    
    for trajectory in S_arr
        local x_coarse_temp, y_coarse_temp, _ = coarse_grained_paths_full(trajectory)
        local t_coarse_temp = x_coarse_temp[end]
        push!(find_max_t_coarse, t_coarse_temp)
    end
    
    max_t_coarse = maximum(find_max_t_coarse)
    
    # Time points for evaluation
    ts = vcat(collect(range(0, T, num_timesteps)))
    
    # Initialize shared arrays for parallel computation
    num_workers = Threads.nthreads()
    
    final_transfer_entropy_XY = SharedArray{Float64}(num_simulations, length(ts))
    final_transfer_entropy_YX = SharedArray{Float64}(num_simulations, length(ts))
    final_mutual_information = SharedArray{Float64}(num_simulations, length(ts))
    final_transfer_entropy_rate_XY_analytical = SharedArray{Float64}(num_simulations, length(ts))
    final_transfer_entropy_rate_YX_analytical = SharedArray{Float64}(num_simulations, length(ts))
    final_mutual_information_rate_analytical = SharedArray{Float64}(num_simulations, length(ts))
    final_transfer_entropy_rate_XY = SharedArray{Float64}(num_simulations, length(ts))
    final_transfer_entropy_rate_YX = SharedArray{Float64}(num_simulations, length(ts))
    final_mutual_information_rate = SharedArray{Float64}(num_simulations, length(ts))
    
    # Process each trajectory in parallel
    Threads.@threads for trajectory_num in 1:length(S_arr)
        trajectory = S_arr[trajectory_num]
        local ts_temp_, path_te_XY_temp, path_te_YX_temp, path_mi_temp, path_te_XY_rate_analytical_temp, path_te_YX_rate_analytical_temp, path_mi_rate_analytical_temp,path_te_XY_rate_temp,path_te_YX_rate_temp,path_mi_rate_temp= calculate_information_metrics_single_trajectory(twocell_system, trajectory, ts, starting_state_tuple)
        # Find indices in ts_temp_ that correspond to times in ts
        ts_indices_in_temp = [findall(x -> x == t, ts_temp_)[1] for t in ts if t ∈ ts_temp_]

        # Extract values at those indices
        path_te_XY = path_te_XY_temp[ts_indices_in_temp]
        path_te_YX = path_te_YX_temp[ts_indices_in_temp]
        path_mi = path_mi_temp[ts_indices_in_temp]
        path_te_XY_rate = path_te_XY_rate_temp[ts_indices_in_temp]
        path_te_YX_rate = path_te_YX_rate_temp[ts_indices_in_temp]
        path_mi_rate = path_mi_rate_temp[ts_indices_in_temp]
        path_te_XY_rate_analytical = path_te_XY_rate_analytical_temp[ts_indices_in_temp]
        path_te_YX_rate_analytical = path_te_YX_rate_analytical_temp[ts_indices_in_temp]
        path_mi_rate_analytical = path_mi_rate_analytical_temp[ts_indices_in_temp]

        final_transfer_entropy_XY[trajectory_num, :] = path_te_XY
        final_transfer_entropy_YX[trajectory_num, :] = path_te_YX
        final_mutual_information[trajectory_num, :] = path_mi
        final_transfer_entropy_rate_XY[trajectory_num, :] = path_te_XY_rate
        final_transfer_entropy_rate_YX[trajectory_num, :] = path_te_YX_rate
        final_mutual_information_rate[trajectory_num, :] = path_mi_rate
        final_transfer_entropy_rate_XY_analytical[trajectory_num, :] = path_te_XY_rate_analytical
        final_transfer_entropy_rate_YX_analytical[trajectory_num, :] = path_te_YX_rate_analytical
        final_mutual_information_rate_analytical[trajectory_num, :] = path_mi_rate_analytical
        # Avoid multiple threads writing to console at same time
        Threads.lock(ReentrantLock()) do
            println("Processed trajectory: ", trajectory_num, 
                    "; XY_transfer_entropy: ", path_te_XY[end],
                    "; YX_transfer_entropy: ", path_te_YX[end],
                    "; Path Mutual Information: ", path_mi[end]
                    )
        end
    end    
    
    #--------------------------------
    #Crete dictionary of results
    results_dict = Dict(
        "S_arr" => S_arr,
        "ts" => ts,
        "terminal_classes" => terminal_classes,
        "final_transfer_entropy_XY" => final_transfer_entropy_XY,
        "final_transfer_entropy_YX" => final_transfer_entropy_YX,
        "final_mutual_information" => final_mutual_information,
        "final_transfer_entropy_rate_XY" => final_transfer_entropy_rate_XY,
        "final_transfer_entropy_rate_YX" => final_transfer_entropy_rate_YX,
        "final_mutual_information_rate" => final_mutual_information_rate,
        "final_transfer_entropy_rate_XY_analytical" => final_transfer_entropy_rate_XY_analytical,
        "final_transfer_entropy_rate_YX_analytical" => final_transfer_entropy_rate_YX_analytical,
        "final_mutual_information_rate_analytical" => final_mutual_information_rate_analytical
    )
    return results_dict

end

function helper_func_cal_metrics_given_indices(indices, full_final_transfer_entropy_XY, full_final_transfer_entropy_YX, full_final_mutual_information, full_final_transfer_entropy_rate_XY, full_final_transfer_entropy_rate_YX, full_final_mutual_information_rate, full_final_transfer_entropy_rate_XY_analytical, full_final_transfer_entropy_rate_YX_analytical, full_final_mutual_information_rate_analytical)
    
    #num simulations is number of relevant indices
    num_simulations = length(indices)

    #First just calculate unconditioned averages
    final_transfer_entropy_XY = full_final_transfer_entropy_XY[indices, :]
    final_transfer_entropy_YX = full_final_transfer_entropy_YX[indices, :]
    final_mutual_information = full_final_mutual_information[indices, :]
    final_transfer_entropy_rate_XY = full_final_transfer_entropy_rate_XY[indices, :]
    final_transfer_entropy_rate_YX = full_final_transfer_entropy_rate_YX[indices, :]
    final_mutual_information_rate = full_final_mutual_information_rate[indices, :]
    final_transfer_entropy_rate_XY_analytical = full_final_transfer_entropy_rate_XY_analytical[indices, :]
    final_transfer_entropy_rate_YX_analytical = full_final_transfer_entropy_rate_YX_analytical[indices, :]
    final_mutual_information_rate_analytical = full_final_mutual_information_rate_analytical[indices, :]

    # Averages
    final_average_transfer_entropy_XY = vec(mean(final_transfer_entropy_XY, dims=1))
    final_average_transfer_entropy_YX = vec(mean(final_transfer_entropy_YX, dims=1))
    final_average_mutual_information = vec(mean(final_mutual_information, dims=1))
    final_average_transfer_entropy_rate_XY = vec(mean(final_transfer_entropy_rate_XY, dims=1))
    final_average_transfer_entropy_rate_YX = vec(mean(final_transfer_entropy_rate_YX, dims=1))
    final_average_mutual_information_rate = vec(mean(final_mutual_information_rate, dims=1))
    final_average_transfer_entropy_rate_XY_analytical = vec(mean(final_transfer_entropy_rate_XY_analytical, dims=1))
    final_average_transfer_entropy_rate_YX_analytical = vec(mean(final_transfer_entropy_rate_YX_analytical, dims=1))
    final_average_mutual_information_rate_analytical = vec(mean(final_mutual_information_rate_analytical, dims=1))
    
    # Standard deviations of the original data
    final_std_transfer_entropy_XY = vec(std(final_transfer_entropy_XY, dims=1))
    final_std_transfer_entropy_YX = vec(std(final_transfer_entropy_YX, dims=1))
    final_std_mutual_information = vec(std(final_mutual_information, dims=1))
    final_std_transfer_entropy_rate_XY = vec(std(final_transfer_entropy_rate_XY, dims=1))
    final_std_transfer_entropy_rate_YX = vec(std(final_transfer_entropy_rate_YX, dims=1))
    final_std_mutual_information_rate = vec(std(final_mutual_information_rate, dims=1))
    final_std_transfer_entropy_rate_XY_analytical = vec(std(final_transfer_entropy_rate_XY_analytical, dims=1))
    final_std_transfer_entropy_rate_YX_analytical = vec(std(final_transfer_entropy_rate_YX_analytical, dims=1))
    final_std_mutual_information_rate_analytical = vec(std(final_mutual_information_rate_analytical, dims=1))

    # Standard errors 
    final_std_error_XY = final_std_transfer_entropy_XY ./ sqrt(num_simulations)
    final_std_error_YX = final_std_transfer_entropy_YX ./ sqrt(num_simulations)
    final_std_error_mutual_information = final_std_mutual_information ./ sqrt(num_simulations)
    final_std_error_transfer_entropy_rate_XY = final_std_transfer_entropy_rate_XY ./ sqrt(num_simulations)
    final_std_error_transfer_entropy_rate_YX = final_std_transfer_entropy_rate_YX ./ sqrt(num_simulations)
    final_std_error_mutual_information_rate = final_std_mutual_information_rate ./ sqrt(num_simulations)
    final_std_error_transfer_entropy_rate_XY_analytical = final_std_transfer_entropy_rate_XY_analytical ./ sqrt(num_simulations)
    final_std_error_transfer_entropy_rate_YX_analytical = final_std_transfer_entropy_rate_YX_analytical ./ sqrt(num_simulations)
    final_std_error_mutual_information_rate_analytical = final_std_mutual_information_rate_analytical ./ sqrt(num_simulations)

    return Dict(
        "avg_transfer_entropy_XY" => final_average_transfer_entropy_XY,
        "avg_transfer_entropy_YX" => final_average_transfer_entropy_YX,
        "avg_mutual_information" => final_average_mutual_information,
        "avg_transfer_entropy_rate_XY" => final_average_transfer_entropy_rate_XY,
        "avg_transfer_entropy_rate_YX" => final_average_transfer_entropy_rate_YX,
        "avg_mutual_information_rate" => final_average_mutual_information_rate,
        "avg_mutual_information_rate_analytical" => final_average_mutual_information_rate_analytical,
        "avg_transfer_entropy_rate_XY_analytical" => final_average_transfer_entropy_rate_XY_analytical,
        "avg_transfer_entropy_rate_YX_analytical" => final_average_transfer_entropy_rate_YX_analytical,
        "std_XY" => final_std_transfer_entropy_XY,
        "std_YX" => final_std_transfer_entropy_YX,
        "std_mutual_information" => final_std_mutual_information,
        "std_transfer_entropy_rate_XY" => final_std_transfer_entropy_rate_XY,
        "std_transfer_entropy_rate_YX" => final_std_transfer_entropy_rate_YX,
        "std_mutual_information_rate" => final_std_mutual_information_rate,
        "std_mutual_information_rate_analytical" => final_std_mutual_information_rate_analytical,
        "std_transfer_entropy_rate_XY_analytical" => final_std_transfer_entropy_rate_XY_analytical,
        "std_transfer_entropy_rate_YX_analytical" => final_std_transfer_entropy_rate_YX_analytical,
        "std_error_XY" => final_std_error_XY,
        "std_error_YX" => final_std_error_YX,
        "std_error_mutual_information" => final_std_error_mutual_information,
        "std_error_transfer_entropy_rate_XY" => final_std_error_transfer_entropy_rate_XY,
        "std_error_transfer_entropy_rate_YX" => final_std_error_transfer_entropy_rate_YX,
        "std_error_mutual_information_rate" => final_std_error_mutual_information_rate,
        "std_error_mutual_information_rate_analytical" => final_std_error_mutual_information_rate_analytical,
        "std_error_transfer_entropy_rate_XY_analytical" => final_std_error_transfer_entropy_rate_XY_analytical,
        "std_error_transfer_entropy_rate_YX_analytical" => final_std_error_transfer_entropy_rate_YX_analytical,
    )
end

function calc_overall_info_metrics(results_dict, num_simulations)
   # First get the arrays
   terminal_classes=results_dict["terminal_classes"]
   final_transfer_entropy_XY=results_dict["final_transfer_entropy_XY"]
   final_transfer_entropy_YX=results_dict["final_transfer_entropy_YX"]
   final_mutual_information=results_dict["final_mutual_information"]
   final_transfer_entropy_rate_XY=results_dict["final_transfer_entropy_rate_XY"]
   final_transfer_entropy_rate_YX=results_dict["final_transfer_entropy_rate_YX"]
   final_mutual_information_rate=results_dict["final_mutual_information_rate"]
   final_transfer_entropy_rate_XY_analytical=results_dict["final_transfer_entropy_rate_XY_analytical"]
   final_transfer_entropy_rate_YX_analytical=results_dict["final_transfer_entropy_rate_YX_analytical"]
   final_mutual_information_rate_analytical=results_dict["final_mutual_information_rate_analytical"]
   
    #First just calculate unconditioned averages
    #create an array of all indices
    all_indices = 1:num_simulations
    dict_info_unconditioned = helper_func_cal_metrics_given_indices(all_indices, final_transfer_entropy_XY, final_transfer_entropy_YX, final_mutual_information, final_transfer_entropy_rate_XY, final_transfer_entropy_rate_YX, final_mutual_information_rate, final_transfer_entropy_rate_XY_analytical, final_transfer_entropy_rate_YX_analytical, final_mutual_information_rate_analytical)

    #Now calculate conditioned averages, first based on sucess of just First Cell
    #First get the indices of the successful trajectories
    success_indices_a=findall(terminal_classes .== 1)   
    success_indices_b=findall(terminal_classes .== 2)
    failed_indices=findall(terminal_classes .== -1)

    dict_info_conditioned_a = helper_func_cal_metrics_given_indices(success_indices_a, final_transfer_entropy_XY, final_transfer_entropy_YX, final_mutual_information, final_transfer_entropy_rate_XY, final_transfer_entropy_rate_YX, final_mutual_information_rate, final_transfer_entropy_rate_XY_analytical, final_transfer_entropy_rate_YX_analytical, final_mutual_information_rate_analytical)
    dict_info_conditioned_b = helper_func_cal_metrics_given_indices(success_indices_b, final_transfer_entropy_XY, final_transfer_entropy_YX, final_mutual_information, final_transfer_entropy_rate_XY, final_transfer_entropy_rate_YX, final_mutual_information_rate, final_transfer_entropy_rate_XY_analytical, final_transfer_entropy_rate_YX_analytical, final_mutual_information_rate_analytical)
    dict_info_conditioned_failed = helper_func_cal_metrics_given_indices(failed_indices, final_transfer_entropy_XY, final_transfer_entropy_YX, final_mutual_information, final_transfer_entropy_rate_XY, final_transfer_entropy_rate_YX, final_mutual_information_rate, final_transfer_entropy_rate_XY_analytical, final_transfer_entropy_rate_YX_analytical, final_mutual_information_rate_analytical)
    
    #return all the dictionaries
    return dict_info_unconditioned, dict_info_conditioned_a, dict_info_conditioned_b, dict_info_conditioned_failed
end

function plot_info_metrics(results_dict, info_dict, file_name, plot_path)
    # Create plots
    # Transfer entropy Y→X

    #get the ts
    ts = results_dict["ts"]

    #create the plot path if it doesn't exist
    mkpath(plot_path)
    # Plot average with error bars
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
    # Add a second ribbon for standard error
    plot!(ts, info_dict["avg_transfer_entropy_XY"],
        ribbon=info_dict["std_error_XY"],
        fillalpha=0.5,
        fillcolor=:red)

    # Add 3 example trajectories
    for i in 1:3
        plot!(ts, results_dict["final_transfer_entropy_XY"][i,:], 
            alpha=0.3,
            color=:blue,
            lw=1)
    end

    savefig(combined_plot_XY, joinpath(plot_path, "transfer_xy_"*file_name*".png"))
    # display(combined_plot_XY)
    
    # Transfer entropy rate Y→X
    combined_plot_XY_rate = plot(ts, info_dict["avg_transfer_entropy_rate_XY"],
        ribbon=info_dict["std_transfer_entropy_rate_XY"],
        fillalpha=0.3,
        fillcolor=:blue,
        xlabel="Time", 
        ylabel="Transfer Entropy Rate",
        title="Transfer Entropy Rate Y→X Across Trajectories",
        lw=2,
        legend=false)
        
    # Add a second ribbon for standard error
    plot!(ts, info_dict["avg_transfer_entropy_rate_XY"],
        ribbon=info_dict["std_error_transfer_entropy_rate_XY"],
        fillalpha=0.5,
        fillcolor=:red)

    # Add 3 example trajectories
    for i in 1:3
        plot!(ts, results_dict["final_transfer_entropy_rate_XY"][i,:],
            alpha=0.3, 
            color=:blue,
            lw=1)
    end
        
    savefig(combined_plot_XY_rate, joinpath(plot_path, "transfer_rate_xy_"*file_name*".png"))
    # display(combined_plot_XY_rate)

    # Transfer entropy rate Y→X for single path
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
    # display(combined_plot_XY_rate_for_single_path)
    
    # Transfer entropy X→Y
    combined_plot_YX = plot(ts, info_dict["avg_transfer_entropy_YX"],
        ribbon=info_dict["std_YX"],
        fillalpha=0.3,
        fillcolor=:blue,
        xlabel="Time",
        ylabel="Transfer Entropy", 
        title="Transfer Entropy X→Y Across Trajectories",
        lw=2,
        legend=false)

    # Add a second ribbon for standard error
    plot!(ts, info_dict["avg_transfer_entropy_YX"],
        ribbon=info_dict["std_error_YX"],
        fillalpha=0.5,
        fillcolor=:red)

    # Add 3 example trajectories
    for i in 1:3
        plot!(ts, results_dict["final_transfer_entropy_YX"][i,:],
            alpha=0.3,
            color=:blue,
            lw=1)
    end
        
    savefig(combined_plot_YX, joinpath(plot_path, "transfer_yx_"*file_name*".png"))
    # display(combined_plot_YX)

    # Transfer entropy rate X→Y
    combined_plot_YX_rate = plot(ts, info_dict["avg_transfer_entropy_rate_YX"],
        ribbon=info_dict["std_transfer_entropy_rate_YX"],
        fillalpha=0.3,
        fillcolor=:blue,
        xlabel="Time",
        ylabel="Transfer Entropy Rate", 
        title="Transfer Entropy Rate X→Y Across Trajectories",
        lw=2,
        legend=false)

    # Add a second ribbon for standard error
    plot!(ts, info_dict["avg_transfer_entropy_rate_YX"],
        ribbon=info_dict["std_error_transfer_entropy_rate_YX"],
        fillalpha=0.5,
        fillcolor=:red)

    # Add 3 example trajectories
    for i in 1:3
        plot!(ts, results_dict["final_transfer_entropy_rate_YX"][i,:],
            alpha=0.3,
            color=:blue,
            lw=1)
    end
        
    savefig(combined_plot_YX_rate, joinpath(plot_path, "transfer_rate_yx_"*file_name*".png"))
    # display(combined_plot_YX_rate)

    # Transfer entropy rate X→Y Analytical
    combined_plot_YX_rate_for_single_path = plot(ts, info_dict["avg_transfer_entropy_rate_YX_analytical"],
        ribbon=info_dict["std_error_transfer_entropy_rate_YX_analytical"],
        fillalpha=0.3,
        fillcolor=:blue,
        xlabel="Time",
        ylabel="Transfer Entropy Rate", 
        title="Analytical Transfer Entropy Rate X→Y Across Trajectories",
        lw=2,
        legend=false)


    savefig(combined_plot_YX_rate_for_single_path, joinpath(plot_path, "transfer_rate_yx_analytical_"*file_name*".png"))
    # display(combined_plot_YX_rate_for_single_path)
    
    # Mutual information
    combined_plot_mutual_information = plot(ts, info_dict["avg_mutual_information"],
        ribbon=info_dict["std_mutual_information"],
        fillalpha=0.3,
        fillcolor=:blue,
        xlabel="Time",
        ylabel="Mutual Information", 
        title="Mutual Information Across Trajectories",
        lw=2,
        legend=false)

    # Add a second ribbon for standard error
    plot!(ts, info_dict["avg_mutual_information"],
        ribbon=info_dict["std_error_mutual_information"],
        fillalpha=0.5,
        fillcolor=:red)

    # Add 3 example trajectories
    for i in 1:3
        plot!(ts, results_dict["final_mutual_information"][i,:],
            alpha=0.3,
            color=:blue,
            lw=1)
    end
        
    savefig(combined_plot_mutual_information, joinpath(plot_path, "mi_"*file_name*".png"))
    # display(combined_plot_mutual_information)
    
    # Mutual information rate
    combined_plot_mutual_information_rate = plot(ts, info_dict["avg_mutual_information_rate"],
        ribbon=info_dict["std_mutual_information_rate"],
        fillalpha=0.3,
        fillcolor=:blue,
        xlabel="Time",
        ylabel="Mutual Information Rate", 
        title="Mutual Information Rate Across Trajectories",
        lw=2,
        legend=false)

    # Add a second ribbon for standard error
    plot!(ts, info_dict["avg_mutual_information_rate"],
        ribbon=info_dict["std_error_mutual_information_rate"],
        fillalpha=0.5,
        fillcolor=:red)

    # Add 3 example trajectories
    for i in 1:3
        plot!(ts, results_dict["final_mutual_information_rate"][i,:],
            alpha=0.3,
            color=:blue,
            lw=1)
    end

    savefig(combined_plot_mutual_information_rate, joinpath(plot_path, "mi_rate_"*file_name*".png"))
    # display(combined_plot_mutual_information_rate)

    # Mutual information rate for single path
    combined_plot_mutual_information_rate_for_single_path = plot(ts, info_dict["avg_mutual_information_rate_analytical"],
        ribbon=info_dict["std_error_mutual_information_rate_analytical"],
        fillalpha=0.5,
        fillcolor=:blue,
        xlabel="Time",
        ylabel="Mutual Information Rate",
        title="Analytical Mutual Information Rate Across Trajectories",
        lw=2,
        legend=false)
        
    savefig(combined_plot_mutual_information_rate_for_single_path, joinpath(plot_path, "mi_rate_single_path_"*file_name*".png"))
    # display(combined_plot_mutual_information_rate_for_single_path)
end