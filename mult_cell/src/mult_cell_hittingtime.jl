using JuMP, Ipopt
using HSL_jll

function hitting_time_mod(Q,targetstates_good,targetstates_bad,startstates,λ)
    #this is for affine expression Q, Q is the rate matrix

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
            println("WRONG")
        end
    end
    model2 = Model(Ipopt.Optimizer)
    set_attribute(model2, "linear_solver", "ma97")
    set_optimizer_attribute(model2, "print_level", 0)
    @variable(model2, 0.0<=T[1:n])
    @objective(model2, Min, sum(T.^2))
    @constraint(model2, A * T == b)

    JuMP.optimize!(model2)

    # Get the hitting times
    T_vals = value.(T)

    # Handle infinite hitting times for unreachable states
    for i in 1:n
        if abs(T_vals[i]) < 1e-8 && i ∉ targetstates
            T_vals[i] = Inf
        end
    end

    if Inf in T_vals
        error("Inf in T_vals")
        #something has gone wrong, this shouldn't happen!
    end
    return T_vals 
end

function hitting_time_mod_give_A(Q, targetstates_good, targetstates_bad, startstates, λ, model)
    n = size(Q,1)
    targetstates = [targetstates_good; targetstates_bad]
    
    # Pre-allocate sparse matrix indices and values
    I = Int[]
    J = Int[]
    V = AffExpr[]
    
    # Copy Q matrix elements
    for i in 1:n, j in 1:n
        if Q[i,j] != 0
            push!(I, i)
            push!(J, j)
            push!(V, Q[i,j])
        end
    end
    
    # Handle target states
    for target_state in targetstates
        mask = I .!= target_state
        I = I[mask]
        J = J[mask]
        V = V[mask]
        
        push!(I, target_state)
        push!(J, target_state)
        push!(V, 1.0)
    end
    
    # Handle zero rows
    for i in 1:n
        if all(Q[i, :] .== 0.0) && i ∉ targetstates
            mask = I .!= i
            I = I[mask]
            J = J[mask]
            V = V[mask]
            
            push!(I, i)
            push!(J, i)
            push!(V, 1.0)
        end
    end
    
    A = sparse(I, J, V, n, n)
    return A
end

function hitting_time_mod_give_b(Q, targetstates_good, targetstates_bad, startstates, λ, model)
    n = size(Q,1)
    b = fill(-1.0, n)
    
    # Handle target states
    for target_state in targetstates_good
        b[target_state] = 0.0
    end
    
    for target_state in targetstates_bad
        b[target_state] = λ
    end
    
    # Handle zero rows
    targetstates = [targetstates_good; targetstates_bad]
    for i in 1:n
        if all(Q[i, :] .== 0.0) && i ∉ targetstates
            b[i] = 0.0
        end
    end
    
    return b
end