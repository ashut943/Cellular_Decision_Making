using JuMP, Ipopt
using HSL_jll
using SparseArrays
using IterativeSolvers


function hitting_prob_mod(Q, targetstates_good, targetstates_bad, startstates, λ)
    n = size(Q, 1)
    A = copy(Q)
    b = zeros(n)
    targetstates = [targetstates_good; targetstates_bad]
    
    # Set up absorbing states
    for target_state in targetstates
        A[target_state, :] .= 0.0
        A[target_state, target_state] = 1.0
        b[target_state] = target_state in targetstates_good ? 0.0 : 1.0
    end
    
    # Handle isolated states
    for i in 1:n
        if all(Q[i, :] .== 0.0) && i ∉ targetstates
            println("WRNG")
            A[i, :] .= 0.0
            A[i, i] = 1.0
            b[i] = 0.0
        end
    end
    
    # Solve the system
    model = Model(Ipopt.Optimizer)
    set_attribute(model, "linear_solver", "ma97")
    set_optimizer_attribute(model, "print_level", 0)
    @variable(model, 0.0 <= h[1:n] <= 1.0)
    @objective(model, Min, sum(h.^2))
    @constraint(model, A * h == b)
    JuMP.optimize!(model)
    # println("hitting prob mod")
    # h_ok = A\b
    
    return value.(h)
end

function hitting_prob_mod_give_A(Q, targetstates_good, targetstates_bad, startstates, λ, model)
    n = size(Q, 1)
    A = copy(Q)
    targetstates = [targetstates_good; targetstates_bad]
    
    # Set up absorbing states
    for target_state in targetstates
        A[target_state, :] .= 0.0
        A[target_state, target_state] = 1.0
    end
    
    # Handle isolated states
    for i in 1:n
        if all(Q[i, :] .== 0.0) && i ∉ targetstates
            A[i, :] .= 0.0
            A[i, i] = 1.0
        end
    end
    
    return A
end

function hitting_prob_mod_give_b(Q, targetstates_good, targetstates_bad, startstates, λ, model)
    n = size(Q, 1)
    b = zeros(n)
    targetstates = [targetstates_good; targetstates_bad]
    
    # Set up absorbing states
    for target_state in targetstates
        b[target_state] = target_state in targetstates_good ? 0.0 : 1.0
    end
    
    # Handle isolated states
    for i in 1:n
        if all(Q[i, :] .== 0.0) && i ∉ targetstates
            b[i] = 0.0
        end
    end
    
    return b
end
