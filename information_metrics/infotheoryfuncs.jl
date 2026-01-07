# ============================================================================
# Continuous Time Information Metric Calculation Functions
# ============================================================================
# This module provides functions for computing information-theoretic metrics
# (transfer entropy, mutual information) on coarse-grained stochastic trajectories.
# ============================================================================

# Import required packages
using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO
using Revise
using CellularDecisions
using ExponentialUtilities
using DifferentialEquations
using Logging

# ============================================================================
# Coarse-Graining Path Utilities
# ============================================================================

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
    append!(new_times, times[change_indices])
    append!(x_new, x[change_indices])
    append!(y_new, y[change_indices])

    return x_new,y_new,new_times,x_poss,y_poss,z_poss
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

# ============================================================================
# Q-Matrix Construction (Q_1 and Q_2)
# ----------------------------------------------------------------------------
# Q_1: Generator matrix for hidden variable dynamics (conditioned on observed)
# Q_2: Jump rate vectors for transitions in the observed variable
# ============================================================================

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
            index1=hidden_var_to_index(z)
            index2=hidden_var_to_index(z_p)
            if index1 != index2
                indexing_1=coarse_grained_index(x,y,z,cell_system)
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
                    index1=hidden_var_y_to_index(y,z)
                    index2=hidden_var_y_to_index(y_p,z_p)
                    if(index1 != index2)
                        indexing_1=coarse_grained_index(x,y,z,cell_system)
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
                    index1=hidden_var_x_to_index(x,z)
                    index2=hidden_var_x_to_index(x_p,z_p)
                    if(index1 != index2)
                        indexing_1=coarse_grained_index(x,y,z,cell_system)
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
    Q_2_vector=zeros(Float64,N_q)
    for z in z_poss
        indexing1=coarse_grained_index(x,y,z,cell_system)
        indexing2=coarse_grained_index(x_p,y_p,z,cell_system)
        index=hidden_var_to_index(z)
        Q_2_vector[index]+=Q[indexing1,indexing2]
    end
    return Q_2_vector
end

function build_vector_Q_2_X(cell_system::CellSystem,x,x_p,y_poss,z_poss)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    N_q=length(z_poss)*length(y_poss)
    Q_2_vector=zeros(Float64,N_q)
    for z in z_poss
        for y in y_poss
            indexing1=coarse_grained_index(x,y,z,cell_system)
            indexing2=coarse_grained_index(x_p,y,z,cell_system)
            index=hidden_var_y_to_index(y,z)
            Q_2_vector[index]+=Q[indexing1,indexing2]
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
    Q_2_vector=zeros(Float64,N_q)
    for z in z_poss
        for x in x_poss
            indexing1=coarse_grained_index(x,y,z,cell_system)
            indexing2=coarse_grained_index(x,y_p,z,cell_system)
            index=hidden_var_x_to_index(x,z)
            Q_2_vector[index]+=Q[indexing1,indexing2]
        end
    end
    return Q_2_vector
end

# ============================================================================
# Trajectory-Wise Q-Matrix Construction
# ----------------------------------------------------------------------------
# Build Q_1 matrices and Q_2 vectors for each time interval along a trajectory
# ============================================================================

function build_matrix_Q_1_XY_trajectory(cell_system::CellSystem, trajectory::StochasticPath, x, y, times, x_poss, y_poss, z_poss)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
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

function build_vector_Q_2_XY_trajectory(cell_system::CellSystem, trajectory::StochasticPath, x, y, times, x_poss, y_poss, z_poss)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    times_total=length(times)
    Q_2_vectors=[]
    N_q=length(z_poss)
    for i in 1:times_total
        x_now=x[i]
        y_now=y[i]
        Q_2_vector=zeros(Float64,N_q)
        for x_next in x_poss
            for y_next in y_poss
                if x_next != x_now || y_next != y_now
                    Q_2_vector_temp=build_vector_Q_2_XY(cell_system,x_now,y_now,x_next,y_next,z_poss)
                    Q_2_vector=Q_2_vector+Q_2_vector_temp
                end 
            end
        end
        push!(Q_2_vectors,Q_2_vector)
    end
    return Q_2_vectors
end

function build_matrix_Q_1_X_trajectory(cell_system::CellSystem, trajectory::StochasticPath, x, y, times, x_poss, y_poss, z_poss, x_coarse, y_coarse, times_coarse)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    times_total=length(times_coarse)
    Q_1_matrices=[]
    for i in 1:times_total
        x_now=x_coarse[i]
        Q_1_matrix=build_matrix_Q_1_X(cell_system,x_now,y_poss,z_poss)
        push!(Q_1_matrices,Q_1_matrix)
    end
    return Q_1_matrices
end

function build_vector_Q_2_X_trajectory(cell_system::CellSystem, trajectory::StochasticPath, x, y, times, x_poss, y_poss, z_poss, x_coarse, y_coarse, times_coarse)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    times_total=length(times_coarse)
    Q_2_vectors=[]
    N_q=length(z_poss)*length(y_poss)
    for i in 1:times_total
        x_now=x_coarse[i]
        Q_2_vector=zeros(Float64,N_q)
        for x_next in x_poss
            if x_next != x_now
                Q_2_vector_temp=build_vector_Q_2_X(cell_system,x_now,x_next,y_poss,z_poss)
                Q_2_vector=Q_2_vector+Q_2_vector_temp
            end
        end
        push!(Q_2_vectors,Q_2_vector)
    end
    return Q_2_vectors
end

function build_matrix_Q_1_Y_trajectory(cell_system::CellSystem, trajectory::StochasticPath, x, y, times, x_poss, y_poss, z_poss, x_coarse, y_coarse, times_coarse)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    times_total=length(times_coarse)
    Q_1_matrices=[]
    for i in 1:times_total
        y_now=y_coarse[i]
        Q_1_matrix=build_matrix_Q_1_Y(cell_system,y_now,x_poss,z_poss)
        push!(Q_1_matrices,Q_1_matrix)
    end
    return Q_1_matrices
end

function build_vector_Q_2_Y_trajectory(cell_system::CellSystem, trajectory::StochasticPath, x, y, times, x_poss, y_poss, z_poss, x_coarse, y_coarse, times_coarse)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    times_total=length(times_coarse)
    Q_2_vectors=[]
    N_q=length(z_poss)*length(x_poss)
    for i in 1:times_total
        y_now=y_coarse[i]
        Q_2_vector=zeros(Float64,N_q)
        for y_next in y_poss
            if y_next != y_now
                Q_2_vector_temp=build_vector_Q_2_Y(cell_system,y_now,y_next,x_poss,z_poss)
                Q_2_vector=Q_2_vector+Q_2_vector_temp
            end
        end
        push!(Q_2_vectors,Q_2_vector)   
    end
    return Q_2_vectors
end

# ============================================================================
# Probability Vector Evolution (Filtering Equations)
# ----------------------------------------------------------------------------
# ODE for evolving conditional probability distributions π(z|observed path)
# and functions for computing initial conditions at each jump time
# these are evolved using the filtering equation (SDE) given in the paper
# ============================================================================

function pi_ode!(dπ, π, p, t)
    Q1, S_z = p
    mul!(dπ, Q1', π)
    s̄ = dot(π, S_z)
    π_sum = sum(π)
    for i in eachindex(π)
        dπ[i] -= π[i]*(S_z[i] - s̄)
    end
    return nothing
end
function build_pi_XY_vector_initial_for_each_interval(cell_system::CellSystem, trajectory::StochasticPath, initial_condition::Vector{Float64}, Q_1_matrices::Vector{Matrix{Float64}}, Q_2_vectors::Vector{Vector{Float64}}, x, y, times, x_poss, y_poss, z_poss)
    #function to calculate the initial condition probability vectors for each interval

    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    N_init=length(z_poss)
    initial_conditions_vector=[zeros(Float64,N_init) for i in 1:length(times)]
    initial_conditions_vector[1]=initial_condition

    prob = ODEProblem(
        pi_ode!,
        zeros(Float64, N_init),
        (0.0, 1.0),
        (zeros(Float64, N_init, N_init), zeros(Float64, N_init))
    )

    for i in 2:length(times)
        Q_1_matrix=Q_1_matrices[i-1]
        Q_2_vector=Q_2_vectors[i-1]
        delta_t = times[i]-times[i-1]
        transpose_vector=vec(initial_conditions_vector[i-1])

        #remake the problem for the ODE part of the SDE in this interval
        prob = remake(prob, 
            u0=transpose_vector,
            tspan=(0.0, delta_t),
            p=(Q_1_matrix, Q_2_vector)
        )
        sol = nothing
        with_logger(NullLogger()) do
            sol = solve(prob, alg_hints = :auto)
        end

        #propagated probability vector to the end of the interval
        pi_vector_prev = vec(sol(delta_t))

        #now need to calculate the jump at the end of the interval
        current_state_x=x[i-1]  
        current_state_y=y[i-1]
        next_state_x=x[i]
        next_state_y=y[i]

        q_now=zeros(Float64, N_init)
        for z in z_poss
            index=hidden_var_to_index(z)
            current_state_index=coarse_grained_index(current_state_x,current_state_y,z,cell_system)
            next_state_index=coarse_grained_index(next_state_x,next_state_y,z,cell_system)
            q_now[index]=Q[current_state_index,next_state_index]
        end

        numerator=pi_vector_prev.*q_now
        denominator=dot(pi_vector_prev,q_now)

        if denominator == 0
            #sanity check to ensure that the denominator is not 0
            error("Denominator is 0 in probability calculation for pi_XY!\n" *
                  "numerator: $numerator\n" * 
                  "denominator: $denominator\n" *
                  "pi_vector_prev: $pi_vector_prev\n" *
                  "q_now: $q_now\n" *
                  "current_state_x: $current_state_x\n" *
                  "current_state_y: $current_state_y\n" *
                  "next_state_x: $next_state_x\n" *
                  "next_state_y: $next_state_y")
        end

        jump_cond_temp=numerator/denominator
        pi_vector_current = jump_cond_temp
        initial_conditions_vector[i]=pi_vector_current
    end
    return initial_conditions_vector
end

function build_pi_XY_vector_for_each_interval(t_all::Vector{Float64},cell_system::CellSystem, trajectory::StochasticPath, starting_state_index::Int64, x, y, times, x_poss, y_poss, z_poss)
    #function to calculate the probability vectors for each interval for any time in the interval

    Q = cell_system.Q_matrix
    N = cell_system.internal_states
    state_dict = cell_system.state_dict
    state_dict_inv = cell_system.state_dict_inv

    Q_1_matrices = build_matrix_Q_1_XY_trajectory(cell_system, trajectory, x, y, times, x_poss, y_poss, z_poss)
    Q_1_matrices=Vector{Matrix{Float64}}(Q_1_matrices)
    Q_2_vectors = build_vector_Q_2_XY_trajectory(cell_system, trajectory, x, y, times, x_poss, y_poss, z_poss)
    Q_2_vectors=Vector{Vector{Float64}}(Q_2_vectors)
    N_init=length(z_poss)

    initial_state_XY=zeros(Float64,N_init)
    initial_state_XY[starting_state_index]=1
    initial_conditions_vector=build_pi_XY_vector_initial_for_each_interval(cell_system,trajectory,initial_state_XY, Q_1_matrices, Q_2_vectors, x, y, times, x_poss, y_poss, z_poss) #calculate the initial condition probability vectors for each interval
    current_interval = -1

    pi_vectors_all=[zeros(Float64, N_init) for i in 1:length(t_all)]

    for (t_index,t) in enumerate(t_all)
        pi_vector = zeros(Float64,N_init)
        for i in 1:length(times)-1
            if t>times[i] && t<=times[i+1]
                current_interval = i
                pi_0_vec = vec(initial_conditions_vector[i])
                prob = ODEProblem(
                    pi_ode!,
                    pi_0_vec,
                    (0.0, t-times[i]),
                    (Q_1_matrices[i], Q_2_vectors[i])
                )   
                sol = nothing
                with_logger(NullLogger()) do
                    sol = solve(prob, alg_hints = :auto)
                end
                pi_vector = vec(sol(t-times[i]))
                break
            end
        end
        if current_interval == -1 && t==times[1]
            pi_vector = initial_conditions_vector[1]
        end
        if current_interval == -1 && t>times[end]
            pi_0_vec = vec(initial_conditions_vector[end])
            prob = ODEProblem(
                pi_ode!,
                pi_0_vec,
                (0.0, t-times[end]),
                (Q_1_matrices[end], Q_2_vectors[end])
            )
            sol = nothing
            with_logger(NullLogger()) do
                sol = solve(prob, alg_hints = :auto)
            end
            pi_vector = vec(sol(t-times[end]))
        end
        pi_vectors_all[t_index]=pi_vector
    end
    
    return pi_vectors_all
end

function build_pi_X_vector_initial_for_each_interval(cell_system::CellSystem, trajectory::StochasticPath, initial_condition::Vector{Float64}, Q_1_matrices::Vector{Matrix{Float64}}, Q_2_vectors::Vector{Vector{Float64}}, x, y, times, x_poss, y_poss, z_poss, x_coarse, y_coarse, times_coarse)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    N_init=length(z_poss)*length(y_poss)
    initial_conditions_vector=[zeros(Float64, N_init) for i in 1:length(times_coarse)]
    initial_conditions_vector[1]=initial_condition

    for i in 2:length(times_coarse)    
        Q_1_matrix=Q_1_matrices[i-1]
        Q_2_vector=Q_2_vectors[i-1]
        delta_t = times_coarse[i]-times_coarse[i-1]
        transpose_vector=vec(initial_conditions_vector[i-1])
        prob = ODEProblem(pi_ode!,
            transpose_vector,
            (0.0, delta_t),
            (Q_1_matrix, Q_2_vector)
        )
        sol = nothing
        with_logger(NullLogger()) do
            sol = solve(prob, alg_hints = :auto)
        end
        pi_vector_prev = vec(sol(delta_t))

        current_state_x=x_coarse[i-1]  
        next_state_x=x_coarse[i]
        
        q_now=zeros(Float64,N_init)
        for z in z_poss
            for y in y_poss
                index=hidden_var_y_to_index(y,z)
                current_state_index=coarse_grained_index(current_state_x,y,z,cell_system)
                next_state_index=coarse_grained_index(next_state_x,y,z,cell_system)
                q_now[index]=Q[current_state_index,next_state_index]
            end
        end
        numerator=pi_vector_prev.*q_now
        denominator=dot(pi_vector_prev,q_now)

        if denominator == 0
            #sanity check to ensure that the denominator is not 0
            error("Denominator is 0 in probability calculation for pi_X!\n" *
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
    return initial_conditions_vector
end

function build_pi_X_vector_for_each_interval(t_all::Vector{Float64},cell_system::CellSystem, trajectory::StochasticPath, starting_state_index::Int64, x, y, times, x_poss, y_poss, z_poss, x_coarse, y_coarse, times_coarse)
    Q = cell_system.Q_matrix
    N = cell_system.internal_states
    state_dict = cell_system.state_dict
    state_dict_inv = cell_system.state_dict_inv
    N_init=length(z_poss)*length(y_poss)

    initial_state_X=zeros(Float64,N_init)
    initial_state_X[starting_state_index]=1
    
    Q_1_matrices = build_matrix_Q_1_X_trajectory(cell_system, trajectory, x, y, times, x_poss, y_poss, z_poss, x_coarse, y_coarse, times_coarse)
    Q_1_matrices=Vector{Matrix{Float64}}(Q_1_matrices)
    Q_2_vectors = build_vector_Q_2_X_trajectory(cell_system, trajectory, x, y, times, x_poss, y_poss, z_poss, x_coarse, y_coarse, times_coarse)
    Q_2_vectors=Vector{Vector{Float64}}(Q_2_vectors)

    initial_conditions_vector=build_pi_X_vector_initial_for_each_interval(cell_system,trajectory,initial_state_X, Q_1_matrices, Q_2_vectors, x, y, times, x_poss, y_poss, z_poss, x_coarse, y_coarse, times_coarse)

    pi_vectors_all=[zeros(Float64, N_init) for i in 1:length(t_all)]
    
    for (t_index,t) in enumerate(t_all)
        pi_vector = zeros(Float64,N_init)
        current_interval = -1
        for i in 1:length(times_coarse)-1
            if t>times_coarse[i] && t<=times_coarse[i+1]
                current_interval = times_coarse[i]
                pi_0_vec = vec(initial_conditions_vector[i])
                prob = ODEProblem(
                    pi_ode!,
                    pi_0_vec,
                    (0.0, t-times_coarse[i]),
                    (Q_1_matrices[i], Q_2_vectors[i])
                )
                sol = nothing
                with_logger(NullLogger()) do
                    sol = solve(prob, alg_hints = :auto)
                end
                pi_vector = vec(sol(t-times_coarse[i]))
                break
            end
        end
        if current_interval == -1 && t==times_coarse[1]
            pi_vector = initial_conditions_vector[1]
        end
        if current_interval == -1 && t>times_coarse[end]
            pi_0_vec = vec(initial_conditions_vector[end])
            prob = ODEProblem(
                pi_ode!,
                pi_0_vec,
                (0.0, t-times_coarse[end]),
                (Q_1_matrices[end], Q_2_vectors[end])
            )
            sol = nothing
            with_logger(NullLogger()) do
                sol = solve(prob, alg_hints = :auto)
            end
            pi_vector = vec(sol(t-times_coarse[end]))
        end
        pi_vectors_all[t_index]=pi_vector
    end
    return pi_vectors_all
end

function build_pi_Y_vector_initial_for_each_interval(cell_system::CellSystem, trajectory::StochasticPath, initial_condition::Vector{Float64}, Q_1_matrices::Vector{Matrix{Float64}}, Q_2_vectors::Vector{Vector{Float64}}, x, y, times, x_poss, y_poss, z_poss, x_coarse, y_coarse, times_coarse)
    Q=cell_system.Q_matrix
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    N_init=length(z_poss)*length(x_poss)

    initial_conditions_vector=[zeros(Float64,N_init) for i in 1:length(times_coarse)]
    initial_conditions_vector[1]=initial_condition

    for i in 2:length(times_coarse)    
        Q_1_matrix=Q_1_matrices[i-1]
        Q_2_vector=Q_2_vectors[i-1]
        delta_t = times_coarse[i]-times_coarse[i-1]
        transpose_vector=vec(initial_conditions_vector[i-1])
        prob = ODEProblem(pi_ode!,
            transpose_vector,
            (0.0, delta_t),
            (Q_1_matrix, Q_2_vector)
        )
        sol = nothing
        with_logger(NullLogger()) do
            sol = solve(prob, alg_hints = :auto)
        end
        pi_vector_prev = vec(sol(delta_t))

        current_state_y=y_coarse[i-1]  
        next_state_y=y_coarse[i]
        
        q_now=zeros(Float64,N_init)
        for z in z_poss
            for x in x_poss
                index=hidden_var_x_to_index(x,z)
                current_state_index=coarse_grained_index(x,current_state_y,z,cell_system)
                next_state_index=coarse_grained_index(x,next_state_y,z,cell_system)
                q_now[index]=Q[current_state_index,next_state_index]
            end
        end
        numerator=pi_vector_prev.*q_now
        denominator=dot(pi_vector_prev,q_now)

        if denominator == 0
            #sanity check to ensure that the denominator is not 0
            error("Denominator is 0 in probability calculation for pi_Y!\n" *
                  "numerator: $numerator\n" * 
                  "denominator: $denominator\n" *
                  "pi_vector_prev: $pi_vector_prev\n" *
                  "q_now: $q_now\n" *
                  "current_state_y: $current_state_y\n" *
                  "next_state_y: $next_state_y")
        end

        jump_cond_temp=numerator/denominator
        pi_vector_current = jump_cond_temp
        initial_conditions_vector[i]=pi_vector_current
    end
    return initial_conditions_vector
end

function build_pi_Y_vector_for_each_interval(t_all::Vector{Float64},cell_system::CellSystem, trajectory::StochasticPath, starting_state_index::Int64, x, y, times, x_poss, y_poss, z_poss, x_coarse, y_coarse, times_coarse)
    Q = cell_system.Q_matrix
    N = cell_system.internal_states
    state_dict = cell_system.state_dict
    state_dict_inv = cell_system.state_dict_inv

    N_init=length(z_poss)*length(x_poss)
    initial_state_Y=zeros(Float64,N_init)
    initial_state_Y[starting_state_index]=1

    Q_1_matrices = build_matrix_Q_1_Y_trajectory(cell_system, trajectory, x, y, times, x_poss, y_poss, z_poss, x_coarse, y_coarse, times_coarse)
    Q_1_matrices=Vector{Matrix{Float64}}(Q_1_matrices)
    Q_2_vectors = build_vector_Q_2_Y_trajectory(cell_system, trajectory, x, y, times, x_poss, y_poss, z_poss, x_coarse, y_coarse, times_coarse)
    Q_2_vectors=Vector{Vector{Float64}}(Q_2_vectors)

    initial_conditions_vector=build_pi_Y_vector_initial_for_each_interval(cell_system,trajectory,initial_state_Y, Q_1_matrices, Q_2_vectors, x, y, times, x_poss, y_poss, z_poss, x_coarse, y_coarse, times_coarse)

    pi_vectors_all=[zeros(Float64, N_init) for i in 1:length(t_all)]

    for (t_index,t) in enumerate(t_all)
        pi_vector = zeros(Float64,N_init)
        current_interval = -1
        for i in 1:length(times_coarse)-1
            if t>times_coarse[i] && t<=times_coarse[i+1]
                current_interval = times_coarse[i]
                pi_0_vec = vec(initial_conditions_vector[i])
                prob = ODEProblem(
                    pi_ode!,
                    pi_0_vec,
                    (0.0, t-times_coarse[i]),
                    (Q_1_matrices[i], Q_2_vectors[i])
                )
                sol = nothing
                with_logger(NullLogger()) do
                    sol = solve(prob, alg_hints = :auto)
                end
                pi_vector = vec(sol(t-times_coarse[i]))
                break
            end
        end
        if current_interval == -1 && t==times_coarse[1]
            pi_vector = initial_conditions_vector[1]
        end
        if current_interval == -1 && t>times_coarse[end]
            pi_0_vec = vec(initial_conditions_vector[end])
            prob = ODEProblem(
                pi_ode!,
                pi_0_vec,
                (0.0, t-times_coarse[end]),
                (Q_1_matrices[end], Q_2_vectors[end])
            )
            sol = nothing
            with_logger(NullLogger()) do
                sol = solve(prob, alg_hints = :auto)
            end
            pi_vector = vec(sol(t-times_coarse[end]))
        end
        pi_vectors_all[t_index]=pi_vector
    end
    return pi_vectors_all
end

# ============================================================================
# Effective Coarse Grained Transition Rate Matrix (W)
# ----------------------------------------------------------------------------
# W_XY: Transition rates conditioned on both X and Y trajectories
# W_X:  Transition rates conditioned on X trajectory only
# W_Y:  Transition rates conditioned on Y trajectory only
# ============================================================================

function W_XY_trajectory(t_all::Vector{Float64},cell_system::CellSystem, trajectory::StochasticPath, starting_state_index_XY::Int64)
    N=cell_system.internal_states
    Q=cell_system.Q_matrix
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv

    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)

    pi_vectors_all=build_pi_XY_vector_for_each_interval(t_all,cell_system,trajectory, starting_state_index_XY, x, y, times, x_poss, y_poss, z_poss)

    x_neighbours, y_neighbours, x_boundary, y_boundary = get_neighbours(cell_system)

    #initialize the transition rate matrices for all times
    W_XY_X_neighbours_all=[Dict{typeof(x_neighbours[1]), Float64}() for i in 1:length(t_all)]
    W_XY_Y_neighbours_all=[Dict{typeof(y_neighbours[1]), Float64}() for i in 1:length(t_all)]

    for (t_idx,t) in enumerate(t_all)
        pi_vector=pi_vectors_all[t_idx]
        
        for (i,x_neighbour_increment) in enumerate(x_neighbours)
            W_XY_X_neighbours_all[t_idx][x_neighbour_increment]=0
        end

        for (i,y_neighbour_increment) in enumerate(y_neighbours)
            W_XY_Y_neighbours_all[t_idx][y_neighbour_increment]=0
        end

        X_curr=-1
        Y_curr=-1
        for i in 1:length(times)-1
            if (t>times[i] && t<=times[i+1])
                X_curr=x[i]
                Y_curr=y[i]
                break
            end
        end
        if t==times[1]
            X_curr=x[1]
            Y_curr=y[1]
        end
        if t>times[end]
            X_curr=x[end]
            Y_curr=y[end]
        end

        for (x_neighbour_index, x_neighbour_increment) in enumerate(x_neighbours)
            X_neighbour=X_curr.+x_neighbour_increment
            if X_neighbour in x_boundary
                W_XY_X_neighbours_all[t_idx][x_neighbour_increment]=0
            else
                for z in z_poss
                    index=hidden_var_to_index(z)
                    pi_XY_zbar=pi_vector[index]
                    curr_index=coarse_grained_index(X_curr,Y_curr,z,cell_system)
                    next_index=coarse_grained_index(X_neighbour,Y_curr,z,cell_system)
                    W_XY_X_neighbours_all[t_idx][x_neighbour_increment]+=pi_XY_zbar*Q[curr_index,next_index]
                end
            end
        end  

        for (y_neighbour_index, y_neighbour_increment) in enumerate(y_neighbours)
            Y_neighbour=Y_curr.+y_neighbour_increment
            if Y_neighbour in y_boundary
                W_XY_Y_neighbours_all[t_idx][y_neighbour_increment]=0
            else
                for z in z_poss
                    index=hidden_var_to_index(z)
                    pi_XY_zbar=pi_vector[index]
                    curr_index=coarse_grained_index(X_curr,Y_curr,z,cell_system)
                    next_index=coarse_grained_index(X_curr,Y_neighbour,z,cell_system)
                    W_XY_Y_neighbours_all[t_idx][y_neighbour_increment]+=pi_XY_zbar*Q[curr_index,next_index]
                end
            end
        end  
    end
    return W_XY_X_neighbours_all, W_XY_Y_neighbours_all
end

function W_X_trajectory(t_all::Vector{Float64},cell_system::CellSystem, trajectory::StochasticPath, starting_state_index_X::Int64)
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    Q=cell_system.Q_matrix
    
    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)
    x_coarse,y_coarse,times_coarse=coarse_grained_paths_x(x,y,times)

    pi_vectors_all=build_pi_X_vector_for_each_interval(t_all,cell_system,trajectory, starting_state_index_X, x, y, times, x_poss, y_poss, z_poss, x_coarse, y_coarse, times_coarse)

    x_neighbours, y_neighbours, x_boundary, y_boundary = get_neighbours(cell_system)

    W_X_X_neighbours_all=[Dict{typeof(x_neighbours[1]), Float64}() for i in 1:length(t_all)]

    for (t_idx,t) in enumerate(t_all)
        pi_vector=pi_vectors_all[t_idx]

        for (i,x_neighbour_increment) in enumerate(x_neighbours)
            W_X_X_neighbours_all[t_idx][x_neighbour_increment]=0
        end
        
        X_curr=-1
        for i in 1:length(times_coarse)-1
            if (t>times_coarse[i] && t<=times_coarse[i+1])
                X_curr=x_coarse[i]
                break
            end
        end
        if t==times_coarse[1]
            X_curr=x_coarse[1]
        end
        if t>times_coarse[end]
            X_curr=x_coarse[end]
        end

        for (x_neighbour_index, x_neighbour_increment) in enumerate(x_neighbours)
            X_neighbour=X_curr.+x_neighbour_increment
            if X_neighbour in x_boundary
                W_X_X_neighbours_all[t_idx][x_neighbour_increment]=0
            else
                for z in z_poss
                    for y in y_poss
                        index=hidden_var_y_to_index(y,z)
                        pi_X_ybar_zbar=pi_vector[index]
                        curr_index=coarse_grained_index(X_curr,y,z,cell_system)
                        next_index=coarse_grained_index(X_neighbour,y,z,cell_system)
                        W_X_X_neighbours_all[t_idx][x_neighbour_increment]+=pi_X_ybar_zbar*Q[curr_index,next_index]
                    end
                end
            end
        end  
    end
    return W_X_X_neighbours_all
end

function W_Y_trajectory(t_all::Vector{Float64},cell_system::CellSystem, trajectory::StochasticPath, starting_state_index_Y::Int64)
    N=cell_system.internal_states
    state_dict=cell_system.state_dict
    state_dict_inv=cell_system.state_dict_inv
    Q=cell_system.Q_matrix

    x, y, times, x_poss, y_poss, z_poss = coarse_grained_paths(trajectory)
    x_coarse,y_coarse,times_coarse=coarse_grained_paths_y(x,y,times)

    pi_vectors_all=build_pi_Y_vector_for_each_interval(t_all,cell_system,trajectory, starting_state_index_Y, x, y, times, x_poss, y_poss, z_poss, x_coarse, y_coarse, times_coarse)
   
    x_neighbours, y_neighbours, x_boundary, y_boundary = get_neighbours(cell_system)
    
    W_Y_Y_neighbours_all=[Dict{typeof(y_neighbours[1]), Float64}() for i in 1:length(t_all)]
    for (t_idx,t) in enumerate(t_all)
        pi_vector=pi_vectors_all[t_idx]

        for (i,y_neighbour_increment) in enumerate(y_neighbours)
            W_Y_Y_neighbours_all[t_idx][y_neighbour_increment]=0
        end
        
        Y_curr=-1
        for i in 1:length(times_coarse)-1
            if (t>times_coarse[i] && t<=times_coarse[i+1])
                Y_curr=y_coarse[i]
                break
            end
        end
        if t==times_coarse[1]
            Y_curr=y_coarse[1]
        end
        if t>times_coarse[end]
            Y_curr=y_coarse[end]
        end

        for (y_neighbour_index, y_neighbour_increment) in enumerate(y_neighbours)
            Y_neighbour=Y_curr.+y_neighbour_increment
            if Y_neighbour in y_boundary
                W_Y_Y_neighbours_all[t_idx][y_neighbour_increment]=0
            else
                for z in z_poss
                    for x in x_poss
                        index=hidden_var_x_to_index(x,z)
                        pi_Y_xbar_zbar=pi_vector[index]
                        curr_index=coarse_grained_index(x,Y_curr,z,cell_system)
                        next_index=coarse_grained_index(x,Y_neighbour,z,cell_system)
                        W_Y_Y_neighbours_all[t_idx][y_neighbour_increment]+=pi_Y_xbar_zbar*Q[curr_index,next_index]
                    end
                end
            end
        end  
    end
    return W_Y_Y_neighbours_all
end

# ============================================================================
# Main Information Metrics Calculation
# ----------------------------------------------------------------------------
# Computes path-wise transfer entropy T(X→Y), T(Y→X), and mutual information
# for a single stochastic trajectory using the filtering approach
# ============================================================================

function calculate_information_metrics_single_trajectory(cell_system, trajectory, ts_full, starting_state_tuple)
    #ts_full is the full time vector for the trajectory
    #starting_state_tuple is the starting state of the trajectory

    # Coarse-grain starting state and compute hidden variable indices for each observation scenario
    local starting_state_tuple_coarse=coarse_grain_tuple(starting_state_tuple)
    local x_initial=starting_state_tuple_coarse[1]
    local y_initial=starting_state_tuple_coarse[2]
    local z_initial=starting_state_tuple_coarse[3]

    #convert the starting state to the index of the hidden variable for the XYZ system
    local starting_state_index_XY=hidden_var_to_index(z_initial)
    local starting_state_index_X=hidden_var_y_to_index(y_initial,z_initial)
    local starting_state_index_Y=hidden_var_x_to_index(x_initial,z_initial)

    local x_neighbours, y_neighbours, x_boundary, y_boundary = get_neighbours(cell_system)

    # Coarse-grain the trajectory into X-only, Y-only, and combined paths with their jump times
    local x_coarse_temp, y_coarse_temp, t_coarse_temp, _, _, _ = coarse_grained_paths(trajectory)
    local x_coarse_x_temp, y_coarse_x_temp, t_coarse_x_temp = coarse_grained_paths_x(x_coarse_temp, y_coarse_temp, t_coarse_temp)
    local x_coarse_y_temp, y_coarse_y_temp, t_coarse_y_temp = coarse_grained_paths_y(x_coarse_temp, y_coarse_temp, t_coarse_temp)
    
    #get all relevant computational times: the times queried alongside the jump times, in one single time vector 
    local ts_temp = unique(sort(vcat(ts_full, t_coarse_temp)))
    ts_temp = Float64.(ts_temp)

    # Identify all jump times
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

    #get the indices of the times in the coarse grained trajectory that correspond to the times in the full trajectory
    local x_jump_times_index=t_coarse_x_index_in_t_coarse[2:end]
    local y_jump_times_index=t_coarse_y_index_in_t_coarse[2:end]

    local x_at_jump_times=[]
    for i in 1:length(x_jump_times_index)
        push!(x_at_jump_times, x_coarse_temp[x_jump_times_index[i]-1]) #the state of x after the previous jump is the state of x at the current jump time due to right continuity
    end
    local x_after_jump_times=x_coarse_temp[x_jump_times_index] #the state of x after the jump

    local y_at_jump_times=[]
    for i in 1:length(y_jump_times_index)
        push!(y_at_jump_times, y_coarse_temp[y_jump_times_index[i]-1])
    end
    local y_after_jump_times=y_coarse_temp[y_jump_times_index]

    #classify the jump times by the neighbour increment that caused them
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
                push!(t_jumps_x_neighbours_index_in_t_coarse[x_neighbour_increment], x_jump_times_index[i]) #get the jump times in the coarse grained trajectory that correspond to the jump times in the full trajectory for each neighbour increment
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

    #get the indices of the jump times in the full time vector
    local t_jumps_x_neighbours_index = Dict{Any, Vector{Int64}}()
    for (x_neighbour_index,x_neighbour_increment) in enumerate(x_neighbours)
        t_jumps_x_neighbours_index[x_neighbour_increment]=findall(t -> t ∈ t_coarse_temp[t_jumps_x_neighbours_index_in_t_coarse[x_neighbour_increment]], ts_temp)
    end

    local t_jumps_y_neighbours_index = Dict{Any, Vector{Int64}}()
    for (y_neighbour_index,y_neighbour_increment) in enumerate(y_neighbours)
        t_jumps_y_neighbours_index[y_neighbour_increment]=findall(t -> t ∈ t_coarse_temp[t_jumps_y_neighbours_index_in_t_coarse[y_neighbour_increment]], ts_temp)
    end
    
    # Compute effective transition rates W conditioned on observing both X and Y (W_XY), only X (W_X), or only Y (W_Y)
    local W_XY_X_neighbours=Dict{Any, Vector{Float64}}()
    local W_XY_Y_neighbours=Dict{Any, Vector{Float64}}() 
    for (x_neighbour_index,x_neighbour_increment) in enumerate(x_neighbours)
        W_XY_X_neighbours[x_neighbour_increment]=zeros(length(ts_temp))
    end
    for (y_neighbour_index,y_neighbour_increment) in enumerate(y_neighbours)
        W_XY_Y_neighbours[y_neighbour_increment]=zeros(length(ts_temp))
    end

    W_XY_X_curr_all, W_XY_Y_curr_all = W_XY_trajectory(ts_temp, cell_system, trajectory, starting_state_index_XY)
    for i in 1:length(ts_temp)
        t=ts_temp[i]
        W_XY_X_curr=W_XY_X_curr_all[i]
        W_XY_Y_curr=W_XY_Y_curr_all[i]
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

    W_X_X_curr_all = W_X_trajectory(ts_temp, cell_system, trajectory, starting_state_index_X)
    for i in 1:length(ts_temp)
        t = ts_temp[i]
        W_X_X_curr = W_X_X_curr_all[i]
        for (x_neighbour_index,x_neighbour_increment) in enumerate(x_neighbours)
            W_X_X_neighbours[x_neighbour_increment][i]=W_X_X_curr[x_neighbour_increment]
        end
    end

    W_Y_Y_curr_all = W_Y_trajectory(ts_temp, cell_system, trajectory, starting_state_index_Y)
    for i in 1:length(ts_temp)
        t = ts_temp[i]
        W_Y_Y_curr = W_Y_Y_curr_all[i]
        for (y_neighbour_index,y_neighbour_increment) in enumerate(y_neighbours)
            W_Y_Y_neighbours[y_neighbour_increment][i]=W_Y_Y_curr[y_neighbour_increment]
        end
    end

    # Compute integrands for the continuous part of transfer entropy: log(W_XY/W_X)*W_XY and W_XY - W_X
    local integrand_X_neighbours_1=Dict{Any, Vector{Float64}}()
    local integrand_X_neighbours_2=Dict{Any, Vector{Float64}}()
    local integrand_Y_neighbours_1=Dict{Any, Vector{Float64}}()
    local integrand_Y_neighbours_2=Dict{Any, Vector{Float64}}()

    for (x_neighbour_index, x_neighbour_increment) in enumerate(x_neighbours)
        integrand_X_neighbours_1[x_neighbour_increment]=zeros(length(ts_temp))
        integrand_X_neighbours_2[x_neighbour_increment]=zeros(length(ts_temp))
    end
    for (y_neighbour_index, y_neighbour_increment) in enumerate(y_neighbours)
        integrand_Y_neighbours_1[y_neighbour_increment]=zeros(length(ts_temp))
        integrand_Y_neighbours_2[y_neighbour_increment]=zeros(length(ts_temp))
    end
    for i in 1:length(ts_temp)
        for (x_neighbour_index, x_neighbour_increment) in enumerate(x_neighbours)
            curr_W_XY_X_neighbours=W_XY_X_neighbours[x_neighbour_increment][i]
            curr_W_X_X_neighbours=W_X_X_neighbours[x_neighbour_increment][i]
            integrand_X_neighbours_2[x_neighbour_increment][i] = curr_W_XY_X_neighbours-curr_W_X_X_neighbours

            #epsilon regularized to avoid log(0) due to floating point arithmetic
            epsil=1e-10
            curr_W_XY_X_neighbours_epsilon=max(curr_W_XY_X_neighbours, epsil)
            curr_W_X_X_neighbours_epsilon=max(curr_W_X_X_neighbours, epsil)
            integrand_X_neighbours_1[x_neighbour_increment][i] = (log.(curr_W_XY_X_neighbours_epsilon) - log.(curr_W_X_X_neighbours_epsilon)) * curr_W_XY_X_neighbours
        end
        for (y_neighbour_index, y_neighbour_increment) in enumerate(y_neighbours)
            curr_W_XY_Y_neighbours=W_XY_Y_neighbours[y_neighbour_increment][i]
            curr_W_Y_Y_neighbours=W_Y_Y_neighbours[y_neighbour_increment][i]
            integrand_Y_neighbours_2[y_neighbour_increment][i] = curr_W_XY_Y_neighbours-curr_W_Y_Y_neighbours

            #epsilon regularized to avoid log(0) due to floating point arithmetic
            epsil=1e-10
            curr_W_XY_Y_neighbours_epsilon=max(curr_W_XY_Y_neighbours, epsil)
            curr_W_Y_Y_neighbours_epsilon=max(curr_W_Y_Y_neighbours, epsil)
            integrand_Y_neighbours_1[y_neighbour_increment][i] = (log.(curr_W_XY_Y_neighbours_epsilon) - log.(curr_W_Y_Y_neighbours_epsilon)) * curr_W_XY_Y_neighbours
        end
    end

    local integrand_neighbourwise_XY=Dict{Any, Vector{Float64}}()
    local integrand_neighbourwise_YX=Dict{Any, Vector{Float64}}()
    for (x_neighbour_index, x_neighbour_increment) in enumerate(x_neighbours)
        integrand_neighbourwise_XY[x_neighbour_increment]=integrand_X_neighbours_1[x_neighbour_increment] - integrand_X_neighbours_2[x_neighbour_increment] #difference of the two integrands for each neighbour increment
    end
    for (y_neighbour_index, y_neighbour_increment) in enumerate(y_neighbours)
        integrand_neighbourwise_YX[y_neighbour_increment]=integrand_Y_neighbours_1[y_neighbour_increment] - integrand_Y_neighbours_2[y_neighbour_increment] #difference of the two integrands for each neighbour increment
    end

    # Sum integrands across all neighbour directions to get total instantaneous transfer entropy rate
    local curr_integrand_transfer_entropy_XY=zeros(Float64, length(ts_temp))
    local curr_integrand_transfer_entropy_YX=zeros(Float64, length(ts_temp))
    for i in 1:length(ts_temp)
        curr_integrand_transfer_entropy_XY[i] = sum(integrand_neighbourwise_XY[x_neighbour_increment][i] for x_neighbour_increment in x_neighbours) #sum of the integrand for each neighbour increment
        curr_integrand_transfer_entropy_YX[i] = sum(integrand_neighbourwise_YX[y_neighbour_increment][i] for y_neighbour_increment in y_neighbours) #sum of the integrand for each neighbour increment
    end
    
    # Compute discrete (jump) contributions: log(W_XY/W_X) evaluated at actual jump times
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
        epsil = 1e-10
        t_jumps_relevant_x_xneighbours=t_jumps_x_neighbours_index[x_neighbour_increment]
        relevant_W_x_xneighbours[x_neighbour_increment]=W_X_X_neighbours[x_neighbour_increment][t_jumps_relevant_x_xneighbours]
        relevant_W_xy_xneighbours[x_neighbour_increment]=W_XY_X_neighbours[x_neighbour_increment][t_jumps_relevant_x_xneighbours]
        log_relevant_W_x_xneighbours[x_neighbour_increment]=log.(max.(relevant_W_x_xneighbours[x_neighbour_increment], epsil))
        log_relevant_W_xy_xneighbours[x_neighbour_increment]=log.(max.(relevant_W_xy_xneighbours[x_neighbour_increment], epsil))
    end
    for (y_neighbour_index, y_neighbour_increment) in enumerate(y_neighbours)
        epsil = 1e-10
        t_jumps_relevant_y_yneighbours=t_jumps_y_neighbours_index[y_neighbour_increment]
        relevant_W_y_yneighbours[y_neighbour_increment]=W_Y_Y_neighbours[y_neighbour_increment][t_jumps_relevant_y_yneighbours]
        relevant_W_xy_yneighbours[y_neighbour_increment]=W_XY_Y_neighbours[y_neighbour_increment][t_jumps_relevant_y_yneighbours]
        log_relevant_W_y_yneighbours[y_neighbour_increment]=log.(max.(relevant_W_y_yneighbours[y_neighbour_increment], epsil))
        log_relevant_W_xy_yneighbours[y_neighbour_increment]=log.(max.(relevant_W_xy_yneighbours[y_neighbour_increment], epsil))
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
                push!(log_relevant_diff_x_xneighbours_temp[x_neighbour_increment], curr_log_relevant_W_xy_xneighbours[i] - curr_log_relevant_W_x_xneighbours[i])
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
                push!(log_relevant_diff_y_yneighbours_temp[y_neighbour_increment], curr_log_relevant_W_xy_yneighbours[i] - curr_log_relevant_W_y_yneighbours[i])
            end
        end
        local log_relevant_diff_y_yneighbours=zeros(length(ts_temp))
        log_relevant_diff_y_yneighbours[t_jumps_relevant_y_yneighbours_index]=log_relevant_diff_y_yneighbours_temp[y_neighbour_increment]
        log_relevant_diff_y_yneighbours_tots[y_neighbour_increment]=log_relevant_diff_y_yneighbours
        log_relevant_diff_y_yneighbours_cumsum[y_neighbour_increment]=cumsum(log_relevant_diff_y_yneighbours)
    end

    # Combine cumulative discrete contributions across all neighbour types
    local discts_parts_xy=zeros(length(ts_temp))
    local discts_parts_yx=zeros(length(ts_temp))
    for (x_neighbour_index,x_neighbour_increment) in enumerate(x_neighbours)
        discts_parts_xy+=log_relevant_diff_x_xneighbours_cumsum[x_neighbour_increment]
    end
    for (y_neighbour_index,y_neighbour_increment) in enumerate(y_neighbours)
        discts_parts_yx+=log_relevant_diff_y_yneighbours_cumsum[y_neighbour_increment]
    end

    # Assemble final outputs: path transfer entropies, mutual information, and their instantaneous rates
    local path_te_XY = discts_parts_xy #this is single trajectory term for T_{Y->X}
    local path_te_XY_rate_for_single_path = curr_integrand_transfer_entropy_XY #this is single trajectory term for dT_{Y->X}/dt
    local path_te_YX = discts_parts_yx #this is single trajectory term for T_{X->Y}
    local path_te_YX_rate_for_single_path = curr_integrand_transfer_entropy_YX #this is single trajectory term for dT_{X->Y}/dt
    local path_mi = path_te_XY + path_te_YX #this is single trajectory term for I_{XY}
    local path_mi_rate_for_single_path = path_te_XY_rate_for_single_path+path_te_YX_rate_for_single_path #this is single trajectory term for dI_{XY}/dt

    return ts_temp, path_te_XY, path_te_YX, path_mi, path_te_XY_rate_for_single_path, path_te_YX_rate_for_single_path, path_mi_rate_for_single_path
end
