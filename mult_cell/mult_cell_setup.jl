using JuMP, LinearAlgebra, Distributions, DataStructures
using Revise
using CellularDecisions_final


#this is old version for M=2 cells, just to check if the new version is correct
function get_transition(u, u_, N)

    ((ua,sa), (ub,sb)) = u
    ((ua_,sa_), (ub_,sb_)) = u_
    
    if ub == ub_ && sa == sa_ && sb == sb_ 
        ua_ == ua + 1 && return N * sa + ua + 1, 1
        ua_ == ua - 1 && return sa * N + ua + 2N, 1
    end
    
    if ua == ua_ && sa == sa_ && sb == sb_
        ub_ == ub + 1 && return N * sb + ub + 1, 2
        ub_ == ub - 1 && return sb * N + ub + 2N, 2
    end
    
    if ua == ua_ && ub == ub_
        if sb == sb_ && sa != sa_
            sa == 0 && sa_ == 1 && return 4N + ub + 1, 2
            sa == 1 && sa_ == 0 && return 5N + 2, 1
        end
        if sa == sa_ && sb != sb_
            sb == 0 && sb_ == 1 && return 4N + ua + 1, 1
            sb == 1 && sb_ == 0 && return 5N + 2, 2
        end
    end
    return nothing, nothing
end


function get_transition_new(states,states_, N,M)
    flags_found=zeros(M)
    for i in 1:M
        curr_cell_now=states[i]
        curr_cell_next=states_[i]
        if curr_cell_now!=curr_cell_next
            flags_found[i]=1
        end
    end
    if sum(flags_found)!=1
        return nothing, nothing #not a valid transition
    end
    #otherwise, we might have a valid transition
    curr_cell_index=findfirst(flags_found.==1)
    curr_cell_now=states[curr_cell_index]
    curr_cell_next=states_[curr_cell_index]

    #now check each case
    #case A: change in the internal state of this cell, while no change in the signalling state
    if curr_cell_now[1]!=curr_cell_next[1] && curr_cell_now[2]==curr_cell_next[2]
        #case A1: increment in the internal state
        curr_cell_next[1]==curr_cell_now[1]+1 && return [curr_cell_now[2]*N+curr_cell_now[1]+1], [curr_cell_index]
        #case A2: decrement in the internal state
        curr_cell_next[1]==curr_cell_now[1]-1 && return [curr_cell_now[2]*N+curr_cell_now[1]+2N], [curr_cell_index]
    end
    #case B: change in the signalling state of this cell, while no change in the internal state
    if curr_cell_now[1]==curr_cell_next[1] && curr_cell_now[2]!=curr_cell_next[2]
        #case B1: original signalling state is 0, new signalling state is 1
        if curr_cell_now[2]==0 && curr_cell_next[2]==1
            return_val_1=[]
            return_val_2=[]
            for j in 1:M
                if j!=curr_cell_index
                    j_cell_now=states[j]
                    j_cell_now_internal=j_cell_now[1]
                    # j_cell_now_signalling=j_cell_now[2]
                    push!(return_val_1, 4*N+j_cell_now_internal+1)
                    push!(return_val_2, j)
                end
            end
            return return_val_1, return_val_2
        end
        #case B2: original signalling state is 1, new signalling state is 0
        if curr_cell_now[2]==1 && curr_cell_next[2]==0
            return_val_1=[5*N+2]
            return_val_2=[curr_cell_index]
            return return_val_1, return_val_2
        end
    end
    return nothing, nothing
end



function Q_maker(P_dict,N,M,statedict,statedictinv)
    ni,_=CellularDecisions_final.varioussizes(N,M)
    Q=zeros(ni,ni)
    for (u, u_) in Iterators.product(values(statedict), values(statedict))
        if u == u_
            continue
        end
        i = statedictinv[u]
        j = statedictinv[u_]
        # tempk, tempk_index = get_transition(u, u_, N)
        wow, wow_index = get_transition_new(u, u_, N,M)
        # if tempk!== nothing
        #     if u[1] != u_[1]
        #         Q[i+1, j+1] = P1[tempk] # transition according to P1
        #     else
        #         Q[i+1, j+1] = P2[tempk] # transition according to P2
        #     end
        # end
        # if tempk !== nothing
        #     if tempk_index == 1
        #         Q[i+1, j+1] = P1[tempk]
        #     else 
        #         Q[i+1, j+1] = P2[tempk]
        #     end
        # end
        if wow !== nothing
            # wow_index_=wow_index[1]
            # Q[i+1, j+1] = P_dict[wow_index_][wow[1]]
            qval=0
            for wow_i_ in 1:length(wow)
                qval+=P_dict[wow_index[wow_i_]][wow[wow_i_]]
            end
            Q[i+1, j+1] = qval
            # if wow_index_ == 1
            #     Q[i+1, j+1] = P1[wow[1]]
            # else 
            #     Q[i+1, j+1] = P2[wow[1]]
            # end
        end
    end
    for i in 1:ni
        Q[i,i] = -sum(Q[i, :])
    end
    return Q
    
end

function Q_maker_original_mod(P_dict, N::Int64, M, model,statedict,statedictinv)
    ni,_=CellularDecisions_final.varioussizes(N,M)
    Q=@expression(model, zeros(AffExpr, ni, ni)) 
    for (u, u_) in Iterators.product(values(statedict), values(statedict))
        if u == u_
            continue
        end
        i = statedictinv[u]
        j = statedictinv[u_]
        # tempk, tempk_index = get_transition(u, u_, N)
        wow, wow_index = get_transition_new(u, u_, N, M)
        # if tempk!== nothing
        #     if u[1] != u_[1]
        #         Q[i+1, j+1] = P1[tempk] # transition according to P1
        #     else
        #         Q[i+1, j+1] = P2[tempk] # transition according to P2
        #     end
        # end
        # if tempk !== nothing
        #     if tempk_index == 1
        #         Q[i+1, j+1] = P1[tempk]
        #     else 
        #         Q[i+1, j+1] = P2[tempk]
        #     end
        # end
        if wow !== nothing
            wow_index_=wow_index[1]
            qval=0
            for wow_i_ in 1:length(wow)
                qval+=P_dict[wow_index[wow_i_]][wow[wow_i_]]
            end
            Q[i+1, j+1] = qval
            # if wow_index_ == 1
            #     Q[i+1, j+1] = P1[wow[1]]
            # else 
            #     Q[i+1, j+1] = P2[wow[1]]
            # end
        end
    end
    for i in 1:ni
        Q[i,i] = -sum(Q[i, :])
    end
    return Q
end

function Q_absorbing_states_maker(Q, absorbing_states)
    for i in absorbing_states
        Q[i, :] .= 0.0
    end
    return Q
end