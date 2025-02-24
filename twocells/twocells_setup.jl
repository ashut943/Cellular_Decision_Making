using JuMP, LinearAlgebra, Distributions, DataStructures

function varioussizes(N)
    ni = 4 * (N + 1)^2
    np = 5 * N + 2
    ns  = ni - 12
    nt = np + ns + 1
    return ni,np,ns,nt
end

function statematrices(N)
    ni,np,ns,nt=varioussizes(N)
    S = Dict()
    T = [];
    TG = [];
    TB = [];
    for ua in 0:N
        for ub in 0:N
            for sa in 0:1
                for sb in 0:1
                    index = (2 * sa + sb) * (N + 1)^2 + (N + 1) * ua + ub
                    S[index] = (ua, ub, sa, sb)
                end
            end
        end
    end
    Skeyer = Dict(value => key for (key, value) in S);
    for ua in 0:N
        for ub in 0:N
            for sa in 0:1
                for sb in 0:1
                    i = (2 * sa + sb) * (N + 1)^2 + (N + 1) * ua + ub
                    if ua == N && ub == 0
                        push!(TG, i)
                        push!(T, i)
                    end
                    if ua == 0 && ub == N
                        push!(TG, i)
                        push!(T, i)
                    end
                    if ua == N && ub == N
                        push!(TB, i)
                        push!(T, i)
                    end
                end
            end
        end
    end
    Tc = [i for i in 0:ni-1 if i ∉ T];
    return S,Skeyer,T,TG,TB,Tc
end

function Q_maker(P,N::Int64,λ::Float64,S,Skeyer)
    ni,np,ns,nt=varioussizes(N)
    Q=zeros(ni,ni)
    for (u, u_) in Iterators.product(values(S), values(S))
        if u == u_
            continue
        end
        # println(u," ",u_)
        i = Skeyer[u]
        j = Skeyer[u_]
        flag = 0
        tempk = 0
        ua, ub, sa, sb = u
        ua_, ub_, sa_, sb_ = u_
    
        if ua_==ua+1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sa + ua + 1
        elseif ua_==ua-1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sa * N + ua + 2 * N 
        elseif ub_==ub+1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sb + ub + 1
        elseif ub_==ub-1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sb * N + ub + 2 * N 
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==0 && sa_==1
            flag = 1
            tempk = 4 * N + ub + 1
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==1 && sa_==0
            flag = 1
            tempk = 5 * N + 2
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==0 && sb_==1
            flag = 1
            tempk = 4 * N + ua + 1
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==1 && sb_==0
            flag = 1
            tempk = 5 * N + 2
        end
        if flag == 1
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

function Q_maker_simplified(P,N::Int64,λ::Float64,S,Skeyer)
    ni,np,ns,nt=varioussizes(N)
    Q=zeros(ni,ni)
    for (u, u_) in Iterators.product(values(S), values(S))
        if u == u_
            continue
        end
        i = Skeyer[u]
        j = Skeyer[u_]
        flag = 0
        tempk = 0
        ua, ub, sa, sb = u
        ua_, ub_, sa_, sb_ = u_
    
        if ua_==ua+1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sa + ua + 1
        elseif ua_==ua-1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sa * N + ua + 2 * N 
        elseif ub_==ub+1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sb + ub + 1+(5*N+2)
        elseif ub_==ub-1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sb * N + ub + 2 * N +(5*N+2)
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==0 && sa_==1
            flag = 1
            tempk = 4 * N + ub + 1
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==1 && sa_==0
            flag = 1
            tempk = 5 * N + 2
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==0 && sb_==1
            flag = 1
            tempk = 4 * N + ua + 1+(5*N+2)
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==1 && sb_==0
            flag = 1
            tempk = 5 * N + 2+(5*N+2)
        end
        if flag == 1
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


function Q_maker_original_mod(P,N::Int64,λ::Float64, model,S,Skeyer)
    ni,np,ns,nt=varioussizes(N)
    Q=@expression(model, zeros(AffExpr, ni, ni)) 
    for (u, u_) in Iterators.product(values(S), values(S))
        if u == u_
            continue
        end
        i = Skeyer[u]
        j = Skeyer[u_]
        flag = 0
        ua, ub, sa, sb = u
        ua_, ub_, sa_, sb_ = u_

        tempk = 0

        if ua_==ua+1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sa + ua + 1
        elseif ua_==ua-1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sa * N + ua + 2 * N 
        elseif ub_==ub+1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sb + ub + 1
        elseif ub_==ub-1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sb * N + ub + 2 * N 
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==0 && sa_==1
            flag = 1
            tempk = 4 * N + ub + 1
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==1 && sa_==0
            flag = 1
            tempk = 5 * N + 2
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==0 && sb_==1
            flag = 1
            tempk = 4 * N + ua + 1
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==1 && sb_==0
            flag = 1
            tempk = 5 * N + 2
        end
        if flag == 1
            Q[i+1, j+1] = P[tempk]
        end
    end
    for i in 1:ni
        qi = sum(Q[i, :])
        Q[i,i] = -qi
    end
    return Q
end

function M_maker(N::Int64,λ::Float64,S,Skeyer)
    ni,np,ns,nt=varioussizes(N)
    M=zeros(Int, ni, ni, np);
    for (u, u_) in Iterators.product(values(S), values(S))
        if u == u_
            continue
        end

        i = Skeyer[u]
        j = Skeyer[u_]
        flag = 0
        ua, ub, sa, sb = u
        ua_, ub_, sa_, sb_ = u_

        tempk = 0

        if ua_==ua+1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sa + ua + 1
        elseif ua_==ua-1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sa * N + ua + 2 * N 
        elseif ub_==ub+1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sb + ub + 1
        elseif ub_==ub-1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sb * N + ub + 2 * N 
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==0 && sa_==1
            flag = 1
            tempk = 4 * N + ub + 1
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==1 && sa_==0
            flag = 1
            tempk = 5 * N + 2
        elseif ua == ua_ && ub == ub_  && sa == sa_ && sb==0 && sb_==1
            flag = 1
            tempk = 4 * N + ua + 1
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==1 && sb_==0
            flag = 1
            tempk = 5 * N + 2
        end

        if flag == 1
            M[i+1, j+1, tempk] = flag
        end
    end
    for k in 1:np
        # Calculate the sum of each row for the kth slice of M (equivalent to M[:,:,k] in Python)
        M_i = [sum(M[:, :, k][i, :]) for i in 1:ni]

        for u in values(S)
            i = Skeyer[u]
            M[i+1, i+1, k] = -M_i[i+1]
        end
    end
    
    return M
end

function Q_maker_using_M(P,N::Int64,λ::Float64,S,Skeyer)
    ni,np,ns,nt=varioussizes(N)
    M=M_maker(N::Int64,λ::Float64,S,Skeyer)
    Q = reduce((x, y) -> x + y, [P[k] * M[:, :, k] for k in 1:np])
    return Q
end

function Q_absorbing_states_maker(Q, absorbing_states)
    for i in absorbing_states
        Q[i, :] .= 0.0
        #Q[i, i] = 1.0
    end
    return Q
end

function M_maker_mod(N::Int64,λ::Float64, model,S,Skeyer)
    ni,np,ns,nt=varioussizes(N)
    M=@expression(model, zeros(AffExpr, ni, ni, np)) 
    for (u, u_) in Iterators.product(values(S), values(S))
        if u == u_
            continue
        end

        i = Skeyer[u]
        j = Skeyer[u_]
        flag = 0
        ua, ub, sa, sb = u
        ua_, ub_, sa_, sb_ = u_

        tempk = 0

        if ua_==ua+1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sa + ua + 1
        elseif ua_==ua-1 && ub == ub_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sa * N + ua + 2 * N 
        elseif ub_==ub+1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = N * sb + ub + 1
        elseif ub_==ub-1 && ua == ua_ && sa == sa_ && sb == sb_
            flag = 1
            tempk = sb * N + ub + 2 * N 
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==0 && sa_==1
            flag = 1
            tempk = 4 * N + ub + 1
        elseif ua == ua_ && ub == ub_ && sb == sb_ && sa==1 && sa_==0
            flag = 1
            tempk = 5 * N + 2
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==0 && sb_==1
            flag = 1
            tempk = 4 * N + ua + 1
        elseif ua == ua_ && ub == ub_ && sa == sa_ && sb==1 && sb_==0
            flag = 1
            tempk = 5 * N + 2
        end

        if flag == 1
            M[i+1, j+1, tempk] = flag
        end
    end
    for k in 1:np
        M_i = [sum(M[:, :, k][i, :]) for i in 1:ni]

        for u in values(S)
            i = Skeyer[u]
            M[i+1, i+1, k] = -M_i[i+1]
        end
    end
    
    return M
end

function Q_maker_using_M_mod(P,N::Int64,λ::Float64,model,S,Skeyer)
    ni,np,ns,nt=varioussizes(N)
    M=M_maker_mod(N,λ,model,S,Skeyer)
    Q = reduce((x, y) -> x + y, [P[k] * M[:, :, k] for k in 1:np])
    return Q
end

function Q_maker_tilde_mod(P,N::Int64,λ::Float64, model,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    Q=Q_maker_original_mod(P,N,λ, model,S,Skeyer)
    R = zeros(ns, ni)
    for i in 1:ns
        R[i,Tc[i]+1] = 1
    end
    Qtilde = R * Q * R'
    return Qtilde
end

function M_maker_tilde(N::Int64,λ::Float64,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    M=M_maker(N,λ,S,Skeyer)
    R = zeros(ns, ni)
    for i in 1:ns
        R[i,Tc[i]+1] = 1
    end
    
    Mtilde=zeros(ns, ns, np)
    for k in 1:np
        Mtilde[:, :, k] = R * M[:, :, k] * R'
    end

    return Mtilde
end

function M_maker_tilde_mod(N::Int64,λ::Float64, model,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    M=M_maker_mod(N,λ,model,S,Skeyer)
    R = zeros(ns, ni)
    for i in 1:ns
        R[i,Tc[i]+1] = 1
    end
    Mtilde=@expression(model, zeros(AffExpr, ns, ns, np)) 
    for k in 1:np
        Mtilde[:, :, k] = R * M[:, :, k] * R'
    end

    return Mtilde
end

function A_maker(N::Int64,λ::Float64,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    Mtilde=M_maker_tilde(N,λ,S,Skeyer,T,TG,TB,Tc)
    Ai = zeros(ns, np, ns)
    for i in 1:ns
        Ai[:,:,i]= Mtilde[i,:,:]   
    end
    return Ai
end

function A_maker_mod(N::Int64,λ::Float64, model,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    Mtilde=M_maker_tilde_mod(N,λ, model,S,Skeyer,T,TG,TB,Tc)
    Ai = @expression(model, zeros(AffExpr, ns, np, ns)) 
    for i in 1:ns
        Ai[:,:,i]= Mtilde[i,:,:]   
    end
    return Ai
end

function alpha_maker(N::Int64,λ::Float64,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    alpha=zeros(np, ns)
    M=M_maker(N,λ,S,Skeyer)
    for i in 1:ns
        Si=Tc[i]
        for j in 1:ni
            for k in 1:np
                if (j-1 ∈ TB)
                    alpha[k,i]+=M[Si+1,j,k]
                end
                
            end
        end
    end
    return alpha
end

function alpha_maker_mod(N::Int64,λ::Float64, model,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    alpha=@expression(model, zeros(AffExpr, np, ns))
    M=M_maker_mod(N,λ,model,S,Skeyer)
    for i in 1:ns
        Si=Tc[i]
        for j in 1:ni
            for k in 1:np
                if (j-1 ∈ TB)
                    alpha[k,i]+=M[Si+1,j,k]
                end
                
            end
        end
    end
    return alpha
end

function D_maker(N::Int64,λ::Float64,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    Di=zeros(nt, nt, ns)

    alpha=alpha_maker(N,λ,S,Skeyer,T,TG,TB,Tc)
    Ai=A_maker(N,λ,S,Skeyer,T,TG,TB,Tc)
    
    for i in 1:ns
        Di[np+1:np+ns,1:np,i]=Ai[:,:,i]
        Di[1:np,nt,i]=λ*alpha[:,i]'
        Di[nt,nt,i]=1
    end
    D_i=copy(Di)
    for i in 1:ns
        D_i[ :, :,i] = (Di[ :, :,i] + Di[ :, :,i]') / 2
    end
    return D_i
end

function E_maker(N::Int64,λ::Float64,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    Ei=zeros(nt, nt, nt)

    alpha=alpha_maker(N,λ,S,Skeyer,T,TG,TB,Tc)
    Ai=A_maker(N,λ,S,Skeyer,T,TG,TB,Tc)
    
    for i in 1:nt
        Ei[i,nt,i]=1
    end
    
    E_i=copy(Ei)
    
    for i in 1:nt
        E_i[ :, :,i] = (Ei[ :, :,i] + Ei[ :, :,i]') / 2
    end
    return E_i
end

function C_maker(N::Int64,λ::Float64,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    C_i=zeros(nt, nt, np)
    
    for i in 1:np
        C_i[i,i,i]=1
    end
    return C_i
end

function F_maker(N::Int64,λ::Float64,S,Skeyer,T,TG,TB,Tc)
    ni,np,ns,nt=varioussizes(N)
    Fij=zeros(nt, nt, np, np)
    for i in 1:np
        for j in 1:np
            Fij[i,j,i,j]=1
        end
    end
    Fiji=copy(Fij)
    for i in 1:np
        for j in 1:np
            Fiji[ :, :,i,j] = (Fij[ :, :,i,j] + Fij[ :, :,i,j]') / 2
        end
    end
    print(size(Fiji))
    return Fiji
end