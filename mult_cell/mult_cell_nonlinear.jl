using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO, DataStructures, MathOptInterface, Printf, MosekTools
using Revise
import CellularDecisions_final

function run_nonlinear_solver(N::Int, M::Int,λ::Float64, initial_state::Int, initial_P_values_dict::Dict{Int, Vector{Float64}}, initial_tauval_array::Vector{Float64}, fix_P_dict::Dict{Int, Bool}, independent_cells::Bool, boundary_type::String)
    #Find upper bound
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 1000)
    set_optimizer_attribute(model, "tol", 1e-8)
    # set_silent(model) 
    # set_optimizer_attribute(model, "print_level", 0)

    state_dict,state_dict_inv,_,TG,TB,Tc=CellularDecisions_final.statematrices(N, M, boundary_type);

    ni,np=CellularDecisions_final.varioussizes(N, M)


    targetstates_good=[target_state+1 for target_state ∈ TG];
    targetstates_bad=[target_state+1 for target_state ∈ TB];
    startstates=[start_state+1 for start_state ∈ Tc];
    allstates=[startstates;targetstates_good; targetstates_bad]
    all_targetstates = vcat(targetstates_good, targetstates_bad)

    # @variable(model, 0<=P1_[1:np] <= 1) 
    # @variable(model, 0<=P2_[1:np] <= 1) 
    @variable(model, 0 <= P_[1:M, 1:np] <= 1)
    P_vars = Dict{Int,Vector{VariableRef}}(j => [P_[j,i] for i in 1:np] for j in 1:M)
    # @variable(model, 0<=P_[1:np]) 
    # println(Q_maker_original_mod(P_, N, λ, model, S, Skeyer))
    @variable(model, τ[1:ni]) 
    @objective(model, Min, τ[initial_state])
    @expression(model, A, hitting_time_mod_give_A(Q_maker_original_mod(P_vars, N, M, model, state_dict, state_dict_inv),  targetstates_good, targetstates_bad, allstates, λ, model))
    @expression(model, b, hitting_time_mod_give_b(Q_maker_original_mod(P_vars, N, M, model, state_dict, state_dict_inv),  targetstates_good, targetstates_bad, allstates, λ, model))
    @constraint(model, A * τ == b)
    for i in 1:np
        for m_ in 1:M
            set_start_value(P_vars[m_][i], initial_P_values_dict[m_][i])
        end
    end
    
    for i in 1:ns
        set_start_value(τ[i], initial_tauval_array[i])   # Initial guess for τ
    end  

    for m_ in 1:M
        if fix_P_dict[m_]==true
            for i in 1:np
                @constraint(model, P_vars[m_][i]==initial_P_values_dict[m_][i])
            end
        end
    end

    if independent_cells==false
        for m_ in 2:M
            for i in 1:np
                @constraint(model, P_vars[m_][i]==P_vars[1][i])
            end
        end
    end

    # if independent_cells==false
    #     for i in 1:np
    #         @constraint(model, P_vars[1][i]==P_vars[2][i])
    #     end
    # end


    if boundary_type=="boundary_2" 
        for m_ in 1:M
            @constraint(model, P_vars[m_][1]==0.0)
            @constraint(model, P_vars[m_][N+1]==0.0)
            @constraint(model, P_vars[m_][2*N+N]==0.0)
            @constraint(model, P_vars[m_][3*N+N]==0.0)  
        end
    end
    # if boundary_type=="boundary_2" 
    #     @constraint(model, P1_[1]==0.0)
    #     @constraint(model, P1_[N+1]==0.0)
    #     @constraint(model, P1_[2*N+N]==0.0)
    #     @constraint(model, P1_[3*N+N]==0.0)  
    #     @constraint(model, P2_[1]==0.0)
    #     @constraint(model, P2_[N+1]==0.0)
    #     @constraint(model, P2_[2*N+N]==0.0)
    #     @constraint(model, P2_[3*N+N]==0.0)  
    # end

    # if  boundary_type=="two_absorbing_boundaries"
    #     @constraint(model, P1_[2*N+N]==0.0)
    #     @constraint(model, P1_[3*N+N]==0.0)  
    #     @constraint(model, P2_[2*N+N]==0.0)
    #     @constraint(model, P2_[3*N+N]==0.0)  
    # end

    JuMP.optimize!(model)
    tau_opt=value.(τ);
    # println("tau_opt^*: ", tau_opt[initial_state])
    tau_opt_tilde = tau_opt[startstates]
    P_opt_=value.(P_)
    P_opt_dict= Dict{Int,Vector{Float64}}(j => P_opt_[j,:] for j in 1:M)
    upper_bound=maximum(tau_opt_tilde)
    upper_bound_tau_0=tau_opt[1]  
    # println("P_opt_dict: ", P_opt_dict)
    return upper_bound_tau_0,upper_bound,tau_opt,P_opt_dict,termination_status(model)
end


# using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO, DataStructures, MathOptInterface, Printf, MosekTools
# using Revise
# import CellularDecisions_final

# function run_nonlinear_solver(N::Int, M::Int,λ::Float64, initial_state::Int, initial_Pval_dict::Dict{Int, Vector{Float64}}, initial_tauval_array::Vector{Float64}, fix_P_dict::Dict{Int,Bool}, independent_cells::Bool, boundary_type::String)
#     #Find upper bound
#     model = Model(Ipopt.Optimizer)
#     set_optimizer_attribute(model, "tol", 1e-8)
#     # set_silent(model) 
#     # set_optimizer_attribute(model, "print_level", 0)

#     state_dict,state_dict_inv,_,TG,TB,Tc=CellularDecisions_final.statematrices(N, M, boundary_type);

#     ni,np=CellularDecisions_final.varioussizes(N, M)


#     targetstates_good=[target_state+1 for target_state ∈ TG];
#     targetstates_bad=[target_state+1 for target_state ∈ TB];
#     startstates=[start_state+1 for start_state ∈ Tc];
#     allstates=[startstates;targetstates_good; targetstates_bad]
#     all_targetstates = vcat(targetstates_good, targetstates_bad)

#     # @variable(model, 0 <= P_[1:M, 1:np] <= 1)
#     # P_vars = Dict{Int,Vector{VariableRef}}(j => [P_[j,i] for i in 1:np] for j in 1:M)
#     @variable(model, 0<=P1_[1:np] <= 1) 
#     @variable(model, 0<=P2_[1:np] <= 1) 
#     # @variable(model, 0<=P_[1:np]) 
#     # println(Q_maker_original_mod(P_, N, λ, model, S, Skeyer))
#     @variable(model, τ[1:ni]) 
#     @objective(model, Min, τ[initial_state])
#     @expression(model, A, hitting_time_mod_give_A(Q_maker_original_mod(Dict(1=>P1_,2=>P2_), N, M, model, state_dict, state_dict_inv),  targetstates_good, targetstates_bad, allstates, λ, model))
#     @expression(model, b, hitting_time_mod_give_b(Q_maker_original_mod(Dict(1=>P1_,2=>P2_), N, M, model, state_dict, state_dict_inv),  targetstates_good, targetstates_bad, allstates, λ, model))
#     @constraint(model, A * τ == b)

#     # for j in 1:M
#     #     for i in 1:np
#     #         set_start_value(P_vars[j][i], initial_Pval_dict[j][i])  # Initial guess for Ps
#     #     end
#     # end
    
#     # for i in 1:ns
#     #     set_start_value(τ[i], initial_tauval_array[i])   # Initial guess for τ
#     # end  

#     # for j in 1:M
#     #     if fix_P_dict[j]==true
#     #         for i in 1:np
#     #             @constraint(model, P_vars[j][i]==initial_Pval_dict[j][i])
#     #         end
#     #     end
#     # end

#     # if independent_cells==false
#     #     for j in 1:M-1
#     #         for k in j+1:M
#     #             if j!=k
#     #                 for i in 1:np
#     #                     @constraint(model, P_vars[j][i]==P_vars[k][i])
#     #                 end
#     #             end
#     #         end
#     #     end
#     # end

#     # if boundary_type=="boundary_2" 
#     #     for j in 1:M
#     #         @constraint(model, P_vars[j][1]==0.0)
#     #         @constraint(model, P_vars[j][N+1]==0.0)
#     #         @constraint(model, P_vars[j][2*N+N]==0.0)
#     #         @constraint(model, P_vars[j][3*N+N]==0.0)  
#     #     end
#     # end

#     # if  boundary_type=="two_absorbing_boundaries"
#     #     @constraint(model, P1_[2*N+N]==0.0)
#     #     @constraint(model, P1_[3*N+N]==0.0)  
#     #     @constraint(model, P2_[2*N+N]==0.0)
#     #     @constraint(model, P2_[3*N+N]==0.0)  
#     # end
#     for i in 1:np
#         set_start_value(P1_[i], initial_Pval_dict[1][i])  # Initial guess for P1
#         set_start_value(P2_[i], initial_Pval_dict[2][i])  # Initial guess for P2
#     end
    
#     for i in 1:ns
#         set_start_value(τ[i], initial_tauval_array[i])   # Initial guess for τ
#     end  

#     if fix_P_dict[1]==true
#         for i in 1:np
#             @constraint(model, P1_[i]==initial_Pval_dict[1][i])
#         end
#     end

#     if fix_P_dict[2]==true
#         for i in 1:np
#             @constraint(model, P2_[i]==initial_Pval_dict[2][i])
#         end
#     end

#     if independent_cells==false
#         for i in 1:np
#             @constraint(model, P1_[i]==P2_[i])
#         end
#     end

#     if boundary_type=="boundary_2" 
#         @constraint(model, P1_[1]==0.0)
#         @constraint(model, P1_[N+1]==0.0)
#         @constraint(model, P1_[2*N+N]==0.0)
#         @constraint(model, P1_[3*N+N]==0.0)  
#         @constraint(model, P2_[1]==0.0)
#         @constraint(model, P2_[N+1]==0.0)
#         @constraint(model, P2_[2*N+N]==0.0)
#         @constraint(model, P2_[3*N+N]==0.0)  
#     end
#     JuMP.optimize!(model)

#     tau_opt=value.(τ);
#     tau_opt_tilde = tau_opt[startstates]
#     P1_opt=value.(P1_);
#     P2_opt=value.(P2_);
#     P_opt_dict= Dict{Int,Vector{Float64}}(1 => P1_opt, 2 => P2_opt)

#     upper_bound=maximum(tau_opt_tilde)
#     upper_bound_tau_0=tau_opt[1]  
#     return upper_bound_tau_0,upper_bound,tau_opt,P_opt_dict,termination_status(model)
# end
