using JuMP, LinearAlgebra, Distributions, DataStructures
using Revise
using CellularDecisions


function get_transition(states, states_, N, M)
    # Pre-allocate flags array
    flags_found = zeros(Int, M)
    changed_cell = 0
    
    # Single pass to find changed cell
    for i in 1:M
        if states[i] != states_[i]
            flags_found[i] = 1
            changed_cell = i
        end
    end
    
    # Early return if no valid transition
    if sum(flags_found) != 1
        return nothing, nothing
    end
    
    curr_cell_now = states[changed_cell]
    curr_cell_next = states_[changed_cell]
    
    # Case A: change in internal state, no change in signalling state
    if curr_cell_now[1] != curr_cell_next[1] && curr_cell_now[2] == curr_cell_next[2]
        if curr_cell_next[1] == curr_cell_now[1] + 1
            return [curr_cell_now[2]*N + curr_cell_now[1] + 1], [changed_cell]
        elseif curr_cell_next[1] == curr_cell_now[1] - 1
            return [curr_cell_now[2]*N + curr_cell_now[1] + 2N], [changed_cell]
        end
    end
    
    # Case B: change in signalling state, no change in internal state
    if curr_cell_now[1] == curr_cell_next[1] && curr_cell_now[2] != curr_cell_next[2]
        if curr_cell_now[2] == 0 && curr_cell_next[2] == 1
            return_val_1 = Int[]
            return_val_2 = Int[]
            sizehint!(return_val_1, M-1)
            sizehint!(return_val_2, M-1)
            
            for j in 1:M
                if j != changed_cell
                    push!(return_val_1, 4*N + states[j][1] + 1)
                    push!(return_val_2, j)
                end
            end
            return return_val_1, return_val_2
        elseif curr_cell_now[2] == 1 && curr_cell_next[2] == 0
            return [5*N + 2], [changed_cell]
        end
    end
    
    return nothing, nothing
end



function Q_maker(P_dict,N,M,statedict,statedictinv)
    ni,_=CellularDecisions.varioussizes(N,M)
    Q=zeros(ni,ni)
    for (u, u_) in Iterators.product(values(statedict), values(statedict))
        if u == u_
            continue
        end
        i = statedictinv[u]
        j = statedictinv[u_]

        wow, wow_index = get_transition(u, u_, N,M)

        if wow !== nothing
            qval=0
            for wow_i_ in 1:length(wow)
                qval+=P_dict[wow_index[wow_i_]][wow[wow_i_]]
            end
            Q[i+1, j+1] = qval
        end
    end
    for i in 1:ni
        Q[i,i] = -sum(Q[i, :])
    end
    return Q
    
end

function Q_maker_original_mod(P_dict, N::Int64, M, model, statedict, statedictinv)
    ni,_ = CellularDecisions.varioussizes(N,M)
    
    # Pre-allocate sparse matrix indices and values
    I = Int[]
    J = Int[]
    V = AffExpr[]
    
    # First pass: collect all non-diagonal elements
    for (u, u_) in Iterators.product(values(statedict), values(statedict))
        if u == u_
            continue
        end
        i = statedictinv[u]
        j = statedictinv[u_]
        wow, wow_index = get_transition(u, u_, N, M)
        
        if wow !== nothing
            qval = 0
            for wow_i_ in 1:length(wow)
                qval += P_dict[wow_index[wow_i_]][wow[wow_i_]]
            end
            push!(I, i+1)
            push!(J, j+1)
            push!(V, qval)
        end
    end
    
    # Second pass: calculate diagonal elements
    row_sums = zeros(AffExpr, ni)
    for idx in 1:length(I)
        row_sums[I[idx]] += V[idx]
    end
    
    # Add diagonal elements
    for i in 1:ni
        push!(I, i)
        push!(J, i)
        push!(V, -row_sums[i])
    end
    
    # Create sparse matrix
    Q = sparse(I, J, V, ni, ni)
    return Q
end

function Q_absorbing_states_maker(Q, absorbing_states)
    for i in absorbing_states
        Q[i, :] .= 0.0
    end
    return Q
end