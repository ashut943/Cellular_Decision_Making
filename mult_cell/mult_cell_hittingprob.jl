using JuMP, Ipopt

function hitting_prob_mod(Q,targetstates_good,targetstates_bad,startstates,λ)
    #ugh for simplicity in writing this up, assume irreducible
    #this is for affexpr Q
    #this is error defined in the text (for some lambda).
    n=size(Q,1)
    A=copy(Q)
    b=zeros(n)
    targetstates=[targetstates_good;targetstates_bad]
    for target_state ∈ targetstates
        A[target_state, :] .= 0.0
        A[target_state, target_state] = 1.0
        if(target_state ∈ targetstates_good)
            b[target_state] = 0.0
        else
            b[target_state] = 1.0
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
    @variable(model2, 0.0 <= h[1:n] <= 1.0)
    @constraint(model2, A * h == b)
    # Solve the model
    JuMP.optimize!(model2)

    # Get the hitting probabilities
    h_vals = value.(h)
    # println(h_vals[1])
    return [h_vals[start_state] for start_state ∈ startstates]
end
