using JuMP, LinearAlgebra, Distributions, DataStructures
using Revise
using CellularDecisions

function get_transition(u, u_, N)
    ((ua,sa), (ub,sb)) = u
    ((ua_,sa_), (ub_,sb_)) = u_
    
    if ub == ub_ && sa == sa_ && sb == sb_
        ua_ == ua + 1 && return N * sa + ua + 1
        ua_ == ua - 1 && return sa * N + ua + 2N
    end
    
    if ua == ua_ && sa == sa_ && sb == sb_
        ub_ == ub + 1 && return N * sb + ub + 1
        ub_ == ub - 1 && return sb * N + ub + 2N
    end
    
    if ua == ua_ && ub == ub_
        if sb == sb_ && sa != sa_
            sa == 0 && sa_ == 1 && return 4N + ub + 1
            sa == 1 && sa_ == 0 && return 5N + 2
        end
        if sa == sa_ && sb != sb_
            sb == 0 && sb_ == 1 && return 4N + ua + 1
            sb == 1 && sb_ == 0 && return 5N + 2
        end
    end
    return nothing
end

function get_transition_simplified(u, u_, N)
    ((ua,sa), (ub,sb)) = u
    ((ua_,sa_), (ub_,sb_)) = u_
    
    if ub == ub_ && sa == sa_ && sb == sb_
        ua_ == ua + 1 && return N * sa + ua + 1
        ua_ == ua - 1 && return sa * N + ua + 2N
    end
    
    if ua == ua_ && sa == sa_ && sb == sb_
        ub_ == ub + 1 && return N * sb + ub + 1 + (5*N+2) 
        ub_ == ub - 1 && return sb * N + ub + 2N + (5*N+2)
    end
    
    if ua == ua_ && ub == ub_
        if sb == sb_ && sa != sa_
            sa == 0 && sa_ == 1 && return 4N + ub + 1
            sa == 1 && sa_ == 0 && return 5N + 2
        end
        if sa == sa_ && sb != sb_
            sb == 0 && sb_ == 1 && return 4N + ua + 1 + (5*N+2)
            sb == 1 && sb_ == 0 && return 5N + 2 + (5*N+2)
        end
    end
    return nothing
end

function Q_maker(P,N,λ,statedict,statedictinv)
    ni,_,_,_=CellularDecisions.varioussizes(N)
    Q=zeros(ni,ni)
    for (u, u_) in Iterators.product(values(statedict), values(statedict))
        if u == u_
            continue
        end
        i = statedictinv[u]
        j = statedictinv[u_]
        if (tempk = get_transition(u, u_, N)) !== nothing
            Q[i+1, j+1] = P[tempk]
        end
    end
    for i in 1:ni
        Q[i,i] = -sum(Q[i, :])
    end
    return Q
    
end

function Q_maker_simplified(P,N::Int64,λ::Float64,statedict,statedictinv)
    ni,_,_,_=CellularDecisions.varioussizes(N)
    Q=zeros(ni,ni)
    for (u, u_) in Iterators.product(values(statedict), values(statedict))
        if u == u_
            continue
        end
        i = statedictinv[u]
        j = statedictinv[u_]
        if (tempk = get_transition_simplified(u, u_, N)) !== nothing
            Q[i+1, j+1] = P[tempk]
        end
    end

    for i in 1:ni
        # Calculate the sum of each row for Qi
        qi = sum(Q[i, :])
        Q[i,i] = -qi
    end
    return Q
    
end

function Q_maker_original_mod(P,N::Int64,λ::Float64, model,statedict,statedictinv)
    ni,_,_,_=CellularDecisions.varioussizes(N)
    Q=@expression(model, zeros(AffExpr, ni, ni)) 
    for (u, u_) in Iterators.product(values(statedict), values(statedict))
        if u == u_
            continue
        end
        i = statedictinv[u]
        j = statedictinv[u_]
        if (tempk = get_transition(u, u_, N)) !== nothing
            Q[i+1, j+1] = P[tempk]
        end
    end
    for i in 1:ni
        Q[i,i] = -sum(Q[i, :])
    end
    return Q
end

function M_maker(N::Int64, λ::Float64, statedict, statedictinv)
    ni, np, _, _ = CellularDecisions.varioussizes(N)
    M = zeros(Int, ni, ni, np)
    
    for (u, u_) in Iterators.product(values(statedict), values(statedict))
        if u == u_
            continue
        end

        i = statedictinv[u]
        j = statedictinv[u_]
        
        if (tempk = get_transition(u, u_, N)) !== nothing
            M[i+1, j+1, tempk] = 1
        end
    end
    for k in 1:np
        row_sums = [sum(M[:, :, k][i, :]) for i in 1:ni]
        for u in values(statedict)
            i = statedictinv[u]
            M[i+1, i+1, k] = -row_sums[i+1]
        end
    end
    return M
end

function Q_maker_using_M(P,N::Int64,λ::Float64,statedict,statedictinv)
    _,np,_,_=CellularDecisions.varioussizes(N)
    M=M_maker(N::Int64,λ::Float64,statedict,statedictinv)
    Q = reduce((x, y) -> x + y, [P[k] * M[:, :, k] for k in 1:np])
    return Q
end

function Q_absorbing_states_maker(Q, absorbing_states)
    for i in absorbing_states
        Q[i, :] .= 0.0
    end
    return Q
end