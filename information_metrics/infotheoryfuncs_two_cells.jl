# Import required packages
using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO
using Revise
using CellularDecisions_final

include("../two_cell_experiments/two_cell_coarse_graining.jl")


#--------------------------------
#--------------------------------
#--------------------------------
#--------------------------------
# Below the functions are more general (so for "any well defined coarse graining" procedure)
#--------------------------------
#--------------------------------
#--------------------------------
#--------------------------------

function coarse_grained_paths(trajectory::StochasticPath)
    x,y,z,x_poss,y_poss,z_poss=coarse_grained_paths_full(trajectory)
    times=trajectory.times
    new_times=[]
    x_new=[]
    y_new=[]

    push!(x_new,x[1])
    push!(y_new,y[1])
    push!(new_times,times[1])

    change_indices = findall(i -> x[i] != x[i-1] || y[i] != y[i-1], 2:length(times)) .+ 1
    # Collect values at those indices
    append!(new_times, times[change_indices])
    append!(x_new, x[change_indices])
    append!(y_new, y[change_indices])

    return x_new,y_new,new_times,x_poss,y_poss,z_poss #only return transition times from coarse graining in x,y
end

function coarse_grained_paths_x(x::Vector{Any}, y::Vector{Any}, times::Vector{Any})
    new_times=[]
    x_new=[]
    y_new=[]
    push!(new_times,times[1])
    push!(x_new,x[1])
    push!(y_new,y[1])
    change_indices = findall(i -> x[i] != x[i-1], 2:length(times)) .+ 1
    append!(new_times, times[change_indices])
    append!(x_new, x[change_indices])
    append!(y_new, y[change_indices])
    return x_new,y_new,new_times
end

function coarse_grained_paths_y(x::Vector{Any}, y::Vector{Any}, times::Vector{Any})
    new_times=[]
    x_new=[]
    y_new=[]
    push!(new_times,times[1])
    push!(x_new,x[1])
    push!(y_new,y[1])
    change_indices = findall(i -> y[i] != y[i-1], 2:length(times)) .+ 1
    append!(new_times, times[change_indices])
    append!(x_new, x[change_indices])
    append!(y_new, y[change_indices])
    return x_new,y_new,new_times
end

function build_matrix_Q_1_XY(cell_system::CellSystem,x,y,z_poss)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    N_q=length(z_poss)
    Q_1_matrix=zeros(Float64,N_q,N_q)
    sums=zeros(Float64,N_q)

    for z in z_poss
        for z_p in z_poss
            index1=hidden_var_to_index(z) #2*s_a+s_b+1
            index2=hidden_var_to_index(z_p)
            if index1 != index2
                indexing_1=coarse_grained_index(x,y,z,cell_system) #state_dict_inv[((u_a,s_a),(u_b,s_b))]+1
                indexing_2=coarse_grained_index(x,y,z_p,cell_system)
                Q_1_matrix[index1,index2]=Q[indexing_1,indexing_2]
                sums[index1]+=Q[indexing_1,indexing_2]
            end
        end
    end
    for z in z_poss
        index1=hidden_var_to_index(z)
        Q_1_matrix[index1,index1]=-sums[index1]
    end
    return Q_1_matrix
end

function build_matrix_Q_1_X(cell_system::CellSystem,x,y_poss,z_poss)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    N_q=length(z_poss)*length(y_poss)
    Q_1_matrix=zeros(Float64,N_q,N_q)
    Q_1_matrix_sums=zeros(Float64,N_q)

    for z in z_poss
        for z_p in z_poss
            for y in y_poss
                for y_p in y_poss
                    index1=hidden_var_y_to_index(y,z) #4*u_b+2*s_a+s_b+1
                    index2=hidden_var_y_to_index(y_p,z_p)
                    if(index1 != index2)
                        indexing_1=coarse_grained_index(x,y,z,cell_system) #state_dict_inv[((u_a,s_a),(u_b,s_b))]+1
                        indexing_2=coarse_grained_index(x,y_p,z_p,cell_system)
                        Q_1_matrix[index1,index2]+=Q[indexing_1,indexing_2]
                        Q_1_matrix_sums[index1]+=Q[indexing_1,indexing_2]
                    end
                end
            end
        end
    end
    for z in z_poss
        for y in y_poss
            index1=hidden_var_y_to_index(y,z)
            Q_1_matrix[index1,index1]=-Q_1_matrix_sums[index1]
        end
    end
    return Q_1_matrix
end

function build_matrix_Q_1_Y(cell_system::CellSystem,y,x_poss,z_poss)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    N_q=length(z_poss)*length(x_poss)
    Q_1_matrix=zeros(Float64,N_q,N_q)
    Q_1_matrix_sums=zeros(Float64,N_q)

    for z in z_poss
        for z_p in z_poss
            for x in x_poss
                for x_p in x_poss
                    index1=hidden_var_x_to_index(x,z) #4*u_a+2*s_a+s_b+1
                    index2=hidden_var_x_to_index(x_p,z_p)
                    if(index1 != index2)
                        indexing_1=coarse_grained_index(x,y,z,cell_system) #state_dict_inv[((u_a,s_a),(u_b,s_b))]+1
                        indexing_2=coarse_grained_index(x_p,y,z_p,cell_system)
                        Q_1_matrix[index1,index2]+=Q[indexing_1,indexing_2]
                        Q_1_matrix_sums[index1]+=Q[indexing_1,indexing_2]
                    end
                end
            end
        end
    end
    for z in z_poss
        for x in x_poss
            index1=hidden_var_x_to_index(x,z)
            Q_1_matrix[index1,index1]=-Q_1_matrix_sums[index1]
        end
    end
    return Q_1_matrix
end

function build_vector_Q_2_XY(cell_system::CellSystem,x,y,x_p,y_p,z_poss)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    N_q=length(z_poss)
    Q_2_vector=zeros(Float64,1,N_q)
    for z in z_poss
        indexing1=coarse_grained_index(x,y,z,cell_system)
        indexing2=coarse_grained_index(x_p,y_p,z,cell_system)
        index=hidden_var_to_index(z) #2s_a+s_b+1
        Q_2_vector[1,index]+=Q[indexing1,indexing2]
    end
    return Q_2_vector
end

function build_vector_Q_2_X(cell_system::CellSystem,x,x_p,y_poss,z_poss)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    N_q=length(z_poss)*length(y_poss)
    Q_2_vector=zeros(Float64,1,N_q)
    for z in z_poss
        for y in y_poss
            indexing1=coarse_grained_index(x,y,z,cell_system)
            indexing2=coarse_grained_index(x_p,y,z,cell_system)
            index=hidden_var_y_to_index(y,z) #4*u_b+s_a*2+s_b+1
            Q_2_vector[1,index]+=Q[indexing1,indexing2]
        end
    end
    return Q_2_vector
end

function build_vector_Q_2_Y(cell_system::CellSystem,y,y_p,x_poss,z_poss)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    N_q=length(z_poss)*length(x_poss)
    Q_2_vector=zeros(Float64,1,N_q)
    for z in z_poss
        for x in x_poss
            indexing1=coarse_grained_index(x,y,z,cell_system)
            indexing2=coarse_grained_index(x,y_p,z,cell_system)
            index=hidden_var_x_to_index(x,z) #4*u_a+s_a*2+s_b+1
            Q_2_vector[1,index]+=Q[indexing1,indexing2]
        end
    end
    return Q_2_vector
end

function build_matrix_Q_1_XY_trajectory(cell_system::CellSystem, trajectory::StochasticPath)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)
    times_total=length(times)
    Q_1_matrices=[]
    for i in 1:times_total
        x_now=x[i]
        y_now=y[i]
        Q_1_matrix=build_matrix_Q_1_XY(cell_system,x_now,y_now,z_poss)
        push!(Q_1_matrices,Q_1_matrix)
    end
    return Q_1_matrices
end

function build_vector_Q_2_XY_trajectory(cell_system::CellSystem, trajectory::StochasticPath)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)
    times_total=length(times)
    Q_2_vectors=[]
    for i in 1:times_total-1
        x_now=x[i]
        y_now=y[i]
        x_next=x[i+1]
        y_next=y[i+1]
        Q_2_vector=build_vector_Q_2_XY(cell_system,x_now,y_now,x_next,y_next,z_poss)
        push!(Q_2_vectors,Q_2_vector)   
    end
    return Q_2_vectors
end

function build_matrix_Q_1_X_trajectory(cell_system::CellSystem, trajectory::StochasticPath)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)
    x_coarse,y_coarse,times_coarse=coarse_grained_paths_x(x,y,times)
    times_total=length(times_coarse)
    Q_1_matrices=[]
    for i in 1:times_total
        x_now=x_coarse[i]
        Q_1_matrix=build_matrix_Q_1_X(cell_system,x_now,y_poss,z_poss)
        push!(Q_1_matrices,Q_1_matrix)
    end
    return Q_1_matrices
end

function build_vector_Q_2_X_trajectory(cell_system::CellSystem, trajectory::StochasticPath)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)
    x_coarse,y_coarse,times_coarse=coarse_grained_paths_x(x,y,times)
    times_total=length(times_coarse)
    Q_2_vectors=[]
    for i in 2:times_total
        x_now=x_coarse[i-1]
        x_next=x_coarse[i]
        Q_2_vector=build_vector_Q_2_X(cell_system,x_now,x_next,y_poss,z_poss)
        push!(Q_2_vectors,Q_2_vector)   
    end
    return Q_2_vectors
end


function build_matrix_Q_1_Y_trajectory(cell_system::CellSystem, trajectory::StochasticPath)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)
    x_coarse,y_coarse,times_coarse=coarse_grained_paths_y(x,y,times)
    times_total=length(times_coarse)
    Q_1_matrices=[]
    for i in 1:times_total
        y_now=y_coarse[i]
        Q_1_matrix=build_matrix_Q_1_Y(cell_system,y_now,x_poss,z_poss)
        push!(Q_1_matrices,Q_1_matrix)
    end
    return Q_1_matrices
end

function build_vector_Q_2_Y_trajectory(cell_system::CellSystem, trajectory::StochasticPath)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)
    x_coarse,y_coarse,times_coarse=coarse_grained_paths_y(x,y,times)
    times_total=length(times_coarse)
    Q_2_vectors=[]
    for i in 2:times_total
        y_now=y_coarse[i-1]
        y_next=y_coarse[i]
        Q_2_vector=build_vector_Q_2_Y(cell_system,y_now,y_next,x_poss,z_poss)
        push!(Q_2_vectors,Q_2_vector)   
    end
    return Q_2_vectors
end

function build_pi_XY_vector_initial_for_each_interval(cell_system::CellSystem, trajectory::StochasticPath, initial_condition::Vector{Float64})
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)
    Q_1_matrices=build_matrix_Q_1_XY_trajectory(cell_system,trajectory)
    N_init=length(z_poss)
    initial_conditions_vector=[zeros(Float64,1,N_init) for i in 1:length(times)]
    initial_conditions_vector[1]=reshape(initial_condition,1,N_init)
    #type of initial_conditions_vector is Vector{Vector{Float64}}
    for i in 2:length(times)
        Q_1_matrix=Q_1_matrices[i-1]
        delta_t = times[i]-times[i-1]
        pi_vector_prev = initial_conditions_vector[i-1]*exp(Q_1_matrix * delta_t)
        current_state_x=x[i-1]  
        current_state_y=y[i-1]
        #next state
        next_state_x=x[i]
        next_state_y=y[i]
        #vector for q_now
        q_now=zeros(Float64,1,N_init)
        for z in z_poss
            index=hidden_var_to_index(z)
            current_state_index=coarse_grained_index(current_state_x,current_state_y,z,cell_system)
            next_state_index=coarse_grained_index(next_state_x,next_state_y,z,cell_system)
            q_now[index]=Q[current_state_index,next_state_index]
        end
        #Q_2_vector is the vector of the jump rates from the current state to the next state
        numerator=pi_vector_prev.*q_now
        denominator=dot(pi_vector_prev,q_now)
        if denominator == 0
            println("-------------UH OH pi_XY!-------------------") 
            println("denominator is 0")
            println("numerator: ", numerator)
            println("denominator: ", denominator)
            println("pi_vector_prev: ", pi_vector_prev)
            println("q_now: ", q_now)
            println("current_state_x: ", current_state_x)
            println("current_state_y: ", current_state_y)
            println("next_state_x: ", next_state_x)
            println("next_state_y: ", next_state_y)
            println("--------------------------------")
        end
        jump_cond_temp=numerator/denominator
        pi_vector_current = jump_cond_temp
        initial_conditions_vector[i]=pi_vector_current
    end

    return initial_conditions_vector
end

function build_pi_XY_vector_for_each_interval(t::Float64,cell_system::CellSystem, trajectory::StochasticPath, starting_state_index::Int64)
    Q = cell_system.Q_matrix
    N = cell_system.internal_states
    state_dict = cell_system.state_dict
    state_dict_inv = cell_system.state_dict_inv
    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)
    Q_1_matrices = build_matrix_Q_1_XY_trajectory(cell_system, trajectory)
    Q_2_vectors = build_vector_Q_2_XY_trajectory(cell_system, trajectory)
    N_init=length(z_poss)
    initial_state_XY=zeros(Float64,N_init)
    initial_state_XY[starting_state_index]=1
    pi_vectors = [zeros(Float64, 1, N_init) for i in 1:length(times)]
    initial_conditions_vector=build_pi_XY_vector_initial_for_each_interval(cell_system,trajectory,initial_state_XY)
    pi_vector = zeros(Float64,1,N_init)
    current_interval = -1
    #find the interval that the time t is in and return the pi_vector for that interval
    # t in which interval (t_i, t_{i+1}]
    for i in 1:length(times)-1
        if t>times[i] && t<=times[i+1]
            current_interval = i
            pi_vector = initial_conditions_vector[i]*exp(Q_1_matrices[i]*(t-times[i]))
            break
        end
    end
    if current_interval == -1 && t==times[1]
        pi_vector = initial_conditions_vector[1]
    end
    #if the time is atleast or after the last interval, return the last pi_vector
    if current_interval == -1 && t>times[end]
        pi_vector = initial_conditions_vector[end]*exp(Q_1_matrices[end]*(t-times[end]))
    end
    return pi_vector
end

function build_pi_X_vector_initial_for_each_interval(cell_system::CellSystem, trajectory::StochasticPath, initial_condition::Vector{Float64})
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)
    x_coarse,y_coarse,times_coarse=coarse_grained_paths_x(x,y,times)
    N_init=length(z_poss)*length(y_poss)
    Q_1_matrices=build_matrix_Q_1_X_trajectory(cell_system,trajectory)
    initial_conditions_vector=[zeros(Float64,1,N_init) for i in 1:length(times_coarse)]
    initial_conditions_vector[1]=reshape(initial_condition,1,N_init)
    #type of initial_conditions_vector is Vector{Vector{Float64}}
    for i in 2:length(times_coarse)    
        Q_1_matrix=Q_1_matrices[i-1]
        delta_t = times[i]-times[i-1]
        pi_vector_prev = initial_conditions_vector[i-1]*exp(Q_1_matrix * delta_t)
        current_state_x=x_coarse[i-1]  
        #next state
        next_state_x=x_coarse[i]
        #vector for q_now
        q_now=zeros(Float64,1,N_init)
        for z in z_poss
            for y in y_poss
                index=hidden_var_y_to_index(y,z)#4*u_b+2*s_a+s_b+1
                current_state_index=coarse_grained_index(current_state_x,y,z,cell_system)
                next_state_index=coarse_grained_index(next_state_x,y,z,cell_system)
                q_now[index]=Q[current_state_index,next_state_index]
            end
        end
        #Q_2_vector is the vector of the jump rates from the current state to the next state
        numerator=pi_vector_prev.*q_now
        denominator=dot(pi_vector_prev,q_now)
        if denominator == 0
            error("Denominator is 0 in probability calculation\n" *
                  "numerator: $numerator\n" * 
                  "denominator: $denominator\n" *
                  "pi_vector_prev: $pi_vector_prev\n" *
                  "q_now: $q_now\n" *
                  "current_state_x: $current_state_x\n" *
                  "next_state_x: $next_state_x")
        end
        jump_cond_temp=numerator/denominator
        pi_vector_current = jump_cond_temp
        initial_conditions_vector[i]=pi_vector_current
    end
    # print("Initial conditions vector X: ",initial_conditions_vector)
    return initial_conditions_vector
end

function build_pi_X_vector_for_each_interval(t::Float64,cell_system::CellSystem, trajectory::StochasticPath, starting_state_index::Int64)
    Q = cell_system.Q_matrix
    # Q=Q_absorbing_states_maker(Q, all_targetstates)
    N = cell_system.internal_states
    state_dict = cell_system.state_dict
    state_dict_inv = cell_system.state_dict_inv
    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)
    x_coarse,y_coarse,times_coarse=coarse_grained_paths_x(x,y,times)
    N_init=length(z_poss)*length(y_poss)
    initial_state_X=zeros(Float64,N_init)
    # println("Length of initial state X: ", length(initial_state_X))
    # println("Length of z_poss: ", length(z_poss))
    # println("Length of y_poss: ", length(y_poss))
    # println("Length of x_poss: ", length(x_poss))
    # println("Starting state index: ", starting_state_index)
    initial_state_X[starting_state_index]=1
    Q_1_matrices = build_matrix_Q_1_X_trajectory(cell_system, trajectory)
    Q_2_vectors = build_vector_Q_2_X_trajectory(cell_system, trajectory)
    initial_conditions_vector=build_pi_X_vector_initial_for_each_interval(cell_system,trajectory,initial_state_X)
    pi_vectors = [zeros(Float64, 1, N_init) for i in 1:length(times_coarse)]
    pi_vector = zeros(Float64,1,N_init)
    current_interval = -1
    
    #find the interval that the time t is in and return the pi_vector for that interval
    for i in 1:length(times_coarse)-1
        if t>=times_coarse[i] && t<times_coarse[i+1]
            current_interval = times_coarse[i]
            pi_vector = initial_conditions_vector[i]*exp(Q_1_matrices[i]*(t-times_coarse[i]))
            break
        end
    end
    #if the time is atleast or after the last interval, return the last pi_vector
    if current_interval == -1 && t>=times_coarse[end]
        pi_vector = initial_conditions_vector[end]*exp(Q_1_matrices[end]*(t-times_coarse[end]))
    end
    #find the interval that the time t is in and return the pi_vector for that interval
    for i in 1:length(times_coarse)-1
        if t>times_coarse[i] && t<=times_coarse[i+1]
            current_interval = times_coarse[i]
            pi_vector = initial_conditions_vector[i]*exp(Q_1_matrices[i]*(t-times_coarse[i]))
            break
        end
    end
    #if the time is atleast or after the last interval, return the last pi_vector
    if current_interval == -1 && t==times_coarse[1]
        pi_vector = initial_conditions_vector[1]
    end
    if current_interval == -1 && t>times_coarse[end]
        pi_vector = initial_conditions_vector[end]*exp(Q_1_matrices[end]*(t-times_coarse[end]))
    end
    return pi_vector
end

function build_pi_Y_vector_initial_for_each_interval(cell_system::CellSystem, trajectory::StochasticPath, initial_condition::Vector{Float64})
    Q=cell_system.Q_matrix
    # Q=Q_absorbing_states_maker(Q, all_targetstates)
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)
    x_coarse,y_coarse,times_coarse=coarse_grained_paths_y(x,y,times)
    N_init=length(z_poss)*length(x_poss)
    Q_1_matrices=build_matrix_Q_1_Y_trajectory(cell_system,trajectory)
    initial_conditions_vector=[zeros(Float64,1,N_init) for i in 1:length(times_coarse)]
    initial_conditions_vector[1]=reshape(initial_condition,1,N_init)
    #type of initial_conditions_vector is Vector{Vector{Float64}}
    for i in 2:length(times_coarse)    
        Q_1_matrix=Q_1_matrices[i-1]
        delta_t = times[i]-times[i-1]
        pi_vector_prev = initial_conditions_vector[i-1]*exp(Q_1_matrix * delta_t)
        current_state_y=y_coarse[i-1]  
        #next state
        next_state_y=y_coarse[i]
        #vector for q_now
        q_now=zeros(Float64,1,N_init)
        for z in z_poss
            for x in x_poss
                index=hidden_var_x_to_index(x,z)#4*u_a+2*s_a+s_b+1
                current_state_index=coarse_grained_index(x,current_state_y,z,cell_system)
                next_state_index=coarse_grained_index(x,next_state_y,z,cell_system)
                q_now[index]=Q[current_state_index,next_state_index]
            end
        end
        #Q_2_vector is the vector of the jump rates from the current state to the next state
        numerator=pi_vector_prev.*q_now
        denominator=dot(pi_vector_prev,q_now)
        if denominator == 0
            println("-------------UH OH pi_Y!-------------------") 
            println("denominator is 0")
            println("numerator: ", numerator)
            println("denominator: ", denominator)
            println("pi_vector_prev: ", pi_vector_prev)
            println("q_now: ", q_now)
            println("current_state_y: ", current_state_y)
            println("next_state_y: ", next_state_y)
            println("--------------------------------")
        end
        jump_cond_temp=numerator/denominator
        pi_vector_current = jump_cond_temp
        initial_conditions_vector[i]=pi_vector_current
    end
    return initial_conditions_vector
end

function build_pi_Y_vector_for_each_interval(t::Float64,cell_system::CellSystem, trajectory::StochasticPath, starting_state_index::Int64)
    Q = cell_system.Q_matrix
    N = cell_system.internal_states

    state_dict = cell_system.state_dict
    state_dict_inv = cell_system.state_dict_inv
    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)
    x_coarse,y_coarse,times_coarse=coarse_grained_paths_y(x,y,times)
    N_init=length(z_poss)*length(x_poss)
    initial_state_Y=zeros(Float64,N_init)
    initial_state_Y[starting_state_index]=1
    Q_1_matrices = build_matrix_Q_1_Y_trajectory(cell_system, trajectory)
    Q_2_vectors = build_vector_Q_2_Y_trajectory(cell_system, trajectory)
    initial_conditions_vector=build_pi_Y_vector_initial_for_each_interval(cell_system,trajectory,initial_state_Y)
    pi_vectors = [zeros(Float64, 1, N_init) for i in 1:length(times_coarse)]
    
    pi_vector = zeros(Float64,1,N_init)
    current_interval = -1
    
    #find the interval that the time t is in and return the pi_vector for that interval
    for i in 1:length(times_coarse)-1
        if t>=times_coarse[i] && t<times_coarse[i+1]
            current_interval = times_coarse[i]
            pi_vector = initial_conditions_vector[i]*exp(Q_1_matrices[i]*(t-times_coarse[i]))
            break
        end
    end
    #if the time is atleast or after the last interval, return the last pi_vector
    if current_interval == -1 && t>=times_coarse[end]
        pi_vector = initial_conditions_vector[end]*exp(Q_1_matrices[end]*(t-times_coarse[end]))
    end
    #find the interval that the time t is in and return the pi_vector for that interval
    for i in 1:length(times_coarse)-1
        if t>times_coarse[i] && t<=times_coarse[i+1]
            current_interval = times_coarse[i]
            pi_vector = initial_conditions_vector[i]*exp(Q_1_matrices[i]*(t-times_coarse[i]))
            break
        end
    end
    #if the time is atleast or after the last interval, return the last pi_vector
    if current_interval == -1 && t==times_coarse[1]
        pi_vector = initial_conditions_vector[1]
    end
    if current_interval == -1 && t>times_coarse[end]
        pi_vector = initial_conditions_vector[end]*exp(Q_1_matrices[end]*(t-times_coarse[end]))
    end
    return pi_vector
end

function W_XY_trajectory(t::Float64,cell_system::CellSystem, trajectory::StochasticPath, starting_state_index_XY::Int64)
    N=cell_system.internal_states
    Q=cell_system.Q_matrix
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)
    pi_vector=build_pi_XY_vector_for_each_interval(t,cell_system,trajectory, starting_state_index_XY)
    x_neighbours, y_neighbours, x_boundary, y_boundary = get_neighbours(cell_system)
    W_XY_X_neighbours=Dict{Any, Float64}()#zeros(Float64,length(x_neighbours))
    for (i,x_neighbour_increment) in enumerate(x_neighbours)
        W_XY_X_neighbours[x_neighbour_increment]=0
    end
    W_XY_Y_neighbours=Dict{Any, Float64}()#zeros(Float64,length(y_neighbours))
    for (i,y_neighbour_increment) in enumerate(y_neighbours)
        W_XY_Y_neighbours[y_neighbour_increment]=0
    end
    X_curr=-1 # -1 means not found yet; X_curr is the left limit at the time t
    Y_curr=-1 # -1 means not found yet; Y_curr is the left limit at the time t

    #if t is in the interval (times[i],times[i+1]] then X_curr=x[i] and Y_curr=y[i]
    for i in 1:length(times)-1
        if (t>times[i] && t<=times[i+1])
            X_curr=x[i]
            Y_curr=y[i]
            break
        end
    end
    # if t is the first time, then X_curr=x[1] and Y_curr=y[1]
    if t==times[1]
        X_curr=x[1]
        Y_curr=y[1]
    end
    # if t is after (strictly) the last time, then X_curr=x[end] and Y_curr=y[end]
    if t>times[end]
        X_curr=x[end]
        Y_curr=y[end]
    end
    for (x_neighbour_index, x_neighbour_increment) in enumerate(x_neighbours)
        X_neighbour=X_curr.+x_neighbour_increment
        # println("X boundary: ", x_boundary)
        # println("X neighbour: ", X_neighbour)
        if X_neighbour in x_boundary
            W_XY_X_neighbours[x_neighbour_increment]=0
        else
            for z in z_poss
                index=hidden_var_to_index(z)
                pi_XY_zbar=pi_vector[index]
                curr_index=coarse_grained_index(X_curr,Y_curr,z,cell_system)#state_dict_inv[((X_curr,s_a),(Y_curr,s_b))]+1
                next_index=coarse_grained_index(X_neighbour,Y_curr,z,cell_system)#state_dict_inv[((X_curr+1,s_a),(Y_curr,s_b))]+1
                W_XY_X_neighbours[x_neighbour_increment]+=pi_XY_zbar*Q[curr_index,next_index]
            end
        end
    end  

    for (y_neighbour_index, y_neighbour_increment) in enumerate(y_neighbours)
        Y_neighbour=Y_curr.+y_neighbour_increment
        if Y_neighbour in y_boundary
            W_XY_Y_neighbours[y_neighbour_increment]=0
        else
            for z in z_poss
                index=hidden_var_to_index(z)
                pi_XY_zbar=pi_vector[index]
                curr_index=coarse_grained_index(X_curr,Y_curr,z,cell_system)#state_dict_inv[((X_curr,s_a),(Y_curr,s_b))]+1
                next_index=coarse_grained_index(X_curr,Y_neighbour,z,cell_system)#state_dict_inv[((X_curr+1,s_a),(Y_curr,s_b))]+1
                W_XY_Y_neighbours[y_neighbour_increment]+=pi_XY_zbar*Q[curr_index,next_index]
            end
        end
    end  

    return W_XY_X_neighbours, W_XY_Y_neighbours
end

function W_X_trajectory(t::Float64,cell_system::CellSystem, trajectory::StochasticPath, starting_state_index_X::Int64)
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    Q=cell_system.Q_matrix
    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)
    x_coarse,y_coarse,times_coarse=coarse_grained_paths_x(x,y,times)
    pi_vector=build_pi_X_vector_for_each_interval(t,cell_system,trajectory, starting_state_index_X)
    x_neighbours, y_neighbours, x_boundary, y_boundary = get_neighbours(cell_system)
    W_X_X_neighbours=Dict{Any, Float64}()#zeros(Float64,length(x_neighbours))
    for (i,x_neighbour_increment) in enumerate(x_neighbours)
        W_X_X_neighbours[x_neighbour_increment]=0
    end
    X_curr=-1 # -1 means not found yet; X_curr is the left limit at the time t
    for i in 1:length(times_coarse)-1
        if (t>times_coarse[i] && t<=times_coarse[i+1])
            X_curr=x_coarse[i]
            break
        end
    end
    # if t is the first time, then X_curr=x[1]
    if t==times_coarse[1]
        X_curr=x_coarse[1]
    end
    # if t is after (strictly) the last time, then X_curr=x[end]
    if t>times_coarse[end]
        X_curr=x_coarse[end]
    end

    for (x_neighbour_index, x_neighbour_increment) in enumerate(x_neighbours)
        X_neighbour=X_curr.+x_neighbour_increment
        if X_neighbour in x_boundary
            W_X_X_neighbours[x_neighbour_increment]=0
        else
            for z in z_poss
                for y in y_poss
                    index=hidden_var_y_to_index(y,z)
                    pi_X_ybar_zbar=pi_vector[index]
                    curr_index=coarse_grained_index(X_curr,y,z,cell_system)#state_dict_inv[((X_curr,s_a),(u_b,s_b))]+1
                    next_index=coarse_grained_index(X_neighbour,y,z,cell_system)#state_dict_inv[((X_curr+1,s_a),(Y_curr,s_b))]+1
                    W_X_X_neighbours[x_neighbour_increment]+=pi_X_ybar_zbar*Q[curr_index,next_index]
                end
            end
        end
    end  
    return W_X_X_neighbours
end

function W_Y_trajectory(t::Float64,cell_system::CellSystem, trajectory::StochasticPath, starting_state_index_Y::Int64)
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    Q=cell_system.Q_matrix
    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)
    x_coarse,y_coarse,times_coarse=coarse_grained_paths_y(x,y,times)
    pi_vector=build_pi_Y_vector_for_each_interval(t,cell_system,trajectory, starting_state_index_Y)
    x_neighbours, y_neighbours, x_boundary, y_boundary = get_neighbours(cell_system)
    W_Y_Y_neighbours=Dict{Any, Float64}()#zeros(Float64,length(y_neighbours))
    for (i,y_neighbour_increment) in enumerate(y_neighbours)
        W_Y_Y_neighbours[y_neighbour_increment]=0
    end
    
    Y_curr=-1 # -1 means not found yet; Y_curr is the left limit at the time t
    for i in 1:length(times_coarse)-1
        if (t>times_coarse[i] && t<=times_coarse[i+1])
            Y_curr=y_coarse[i]
            break
        end
    end
    # if t is the first time, then X_curr=x[1]
    if t==times_coarse[1]
        Y_curr=y_coarse[1]
    end
    # if t is after (strictly) the last time, then X_curr=x[end]
    if t>times_coarse[end]
        Y_curr=y_coarse[end]
    end

    for (y_neighbour_index, y_neighbour_increment) in enumerate(y_neighbours)
        Y_neighbour=Y_curr.+y_neighbour_increment
        if Y_neighbour in y_boundary
            W_Y_Y_neighbours[y_neighbour_increment]=0
        else
            for z in z_poss
                for x in x_poss
                    index=hidden_var_x_to_index(x,z)
                    pi_Y_xbar_zbar=pi_vector[index]
                    curr_index=coarse_grained_index(x,Y_curr,z,cell_system)#state_dict_inv[((u_a,s_a),(Y_curr,s_b))]+1
                    next_index=coarse_grained_index(x,Y_neighbour,z,cell_system)#state_dict_inv[((u_a,s_a),(Y_curr,s_b))]+1
                    W_Y_Y_neighbours[y_neighbour_increment]+=pi_Y_xbar_zbar*Q[curr_index,next_index]
                end
            end
        end
    end  
    return W_Y_Y_neighbours
end
