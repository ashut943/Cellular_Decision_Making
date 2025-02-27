using JuMP, Ipopt

function hitting_time_mod(Q,targetstates_good,targetstates_bad,startstates,λ)
    #ugh for simplicity in writing this up, assume irreducible
    #this is for affexpr Q
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
    model2 = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model2, "print_level", 0)
    @variable(model2, T[1:n])
    @constraint(model2, A * T == b)
    # Solve the model
    JuMP.optimize!(model2)

    # Get the hitting times
    T_vals = value.(T)
    # Handle infinite hitting times for unreachable states
    for i in 1:n
        if T_vals[i]==0 && i ∉ targetstates
            T_vals[i] = Inf
        end
    end
    println(T_vals[1])
    return [T_vals[start_state] for start_state ∈ startstates]
end

function hitting_time_give_A(Q,targetstates_good,targetstates_bad,startstates,λ)
    #ugh for simplicity in writing this up, assume irreducible
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
    return A
end

function hitting_time_mod_give_A(Q,targetstates_good,targetstates_bad,startstates,λ,model)
    #ugh for simplicity in writing this up, assume irreducible
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
    return A
end

function hitting_time_give_b(Q,targetstates_good,targetstates_bad,startstates,λ)
    #ugh for simplicity in writing this up, assume irreducible
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
    return b
end

function hitting_time_mod_give_b(Q,targetstates_good,targetstates_bad,startstates,λ,model)
    #ugh for simplicity in writing this up, assume irreducible
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
    return b
end