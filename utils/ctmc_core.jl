using Plots, Printf, LinearAlgebra, Distributions, LightGraphs

function is_irreducible(Q::Matrix{Float64})
    n = size(Q, 1)
    G = SimpleDiGraph(n) 
    for i in 1:n
        for j in 1:n
            if i != j && Q[i, j] > 0
                add_edge!(G, i, j)
            end
        end
    end
    return is_strongly_connected(G)
end

function hitting_time(Q,targetstates_good,targetstates_bad,startstates,λ)
    n=size(Q,1)
    A=copy(Q)
    b=-ones(n)
    targetstates=[targetstates_good;targetstates_bad]
    for target_state ∈ targetstates
        A[target_state, :] .= 0.0
        A[target_state, target_state] = 1.0
        if(target_state ∈ targetstates_good)
            b[target_state] = 0.0
        else
            b[target_state] = λ
        end
    end
    for i in 1:n
        if all(Q[i, :] .== 0.0) && i ∉ targetstates
            A[i, :] .= 0.0
            A[i, i] = 1.0
            b[i] = 0.0
        end
    end
    T = A \ b
    for i in 1:n
        if T[i]==0 && i ∉ targetstates
            T[i]=Inf 
        end
    end
    return [T[start_state] for start_state ∈ startstates]
end

function simulate_mul_ctmc(Q::Array{Float64,2}, initial_state::Int, T::Float64, num_paths::Int, num_time_points::Int)
    num_states = size(Q, 1)
    time_points = range(0, T, length=num_time_points)
    #here to make plotting feasible
    state_counts = zeros(Float64, num_states, num_time_points)
    
    for path in 1:num_paths
        times, states = simulate_ctmc(Q, initial_state, T)
        idx = 1
        for t_idx in 1:num_time_points
            t = time_points[t_idx]
            while idx < length(times) && times[idx+1] <= t
                #essentially check if the current checkpoint time is before or after the latest transition
                idx += 1
            end
            state = states[idx]
            state_counts[state, t_idx] += 1
        end
    end

    state_probs = state_counts / num_paths

    return time_points, state_probs
end
