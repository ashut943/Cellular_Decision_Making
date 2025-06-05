# Import required packages
using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO
using Revise
using CellularDecisions_final

#--------------------------------
#coarse graining with x=u_a, y=u_b, z=(s_a,s_b)
function coarse_grained_paths_full(trajectory::StochasticPath)
    #get the path Unpacked
    trajectory_unpacked=unpack(trajectory)
    #get the variables inside
    u_dict=trajectory_unpacked.u_dict
    s_dict=trajectory_unpacked.s_dict
    u1=u_dict[1]
    u2=u_dict[2]
    u3=u_dict[3]
    s1=s_dict[1]
    s2=s_dict[2]
    s3=s_dict[3]
    #get the coarse grained path
    x=[(u1[i],u2[i]) for i in 1:length(u1)]
    y=u3
    z=[(s1[i],s2[i],s3[i]) for i in 1:length(s1)]
    #return the coarse grained path
    x_poss=[]
    #generate z_poss
    for u_poss1 in collect(0:N)
        for u_poss2 in collect(0:N)
            push!(x_poss, (u_poss1, u_poss2))
        end
    end
    y_poss=collect(0:N)
    z_poss=[]
    # generate z_poss
    s_poss=collect(0:1)
    for s1 in s_poss
        for s2 in s_poss
            for s3 in s_poss
                push!(z_poss, (s1,s2,s3))
            end
        end
    end
    #also return the dictionary of the possible states x,y,z
    return x,y,z,x_poss,y_poss,z_poss
end

function coarse_grain_tuple(tuple_of_states)
    u1=tuple_of_states[1][1]
    u2=tuple_of_states[2][1]
    u3=tuple_of_states[3][1]
    s1=tuple_of_states[1][2]
    s2=tuple_of_states[2][2]
    s3=tuple_of_states[3][2]
    x=(u1,u2)
    y=u3
    z=(s1,s2,s3)
    return (x,y,z)
end

function coarse_grained_index(x,y,z, cell_system::CellSystem)
    state_dict_inv=cell_system.state_dict_inv
    return state_dict_inv[((x[1],z[1]),(x[2],z[2]),(y,z[3]))]+1
end

function hidden_var_to_index(z)
    return 4*z[1]+2*z[2]+z[3]+1
end

function hidden_var_y_to_index(y,z)
    return 8*y+(4*z[1]+2*z[2]+z[3])+1
end

function hidden_var_x_to_index(x,z)
    return 8*((4)*x[1]+x[2])+(4*z[1]+2*z[2]+z[3])+1
end

function get_neighbours(cell_system::CellSystem)
    N=cell_system.internal_states
    x_1_neighbours=[-1,1]
    x_2_neighbours=[-1,1]
    x_neighbours=[(0,-1),(0,1),(-1,0),(1,0)]
    y_neighbours=[-1,1]
    x_1_boundary=[-1,N+1]
    x_2_boundary=[-1,N+1]
    x_boundary=[]
    for u in collect(0:N)
        push!(x_boundary, (u,-1))
        push!(x_boundary, (u,N+1))
        push!(x_boundary, (-1,u)) 
        push!(x_boundary, (N+1,u))
    end
    # push!(x_boundary, (0,-1))
    # push!(x_boundary, (0,N+1))
    # push!(x_boundary, (N,-1))
    # push!(x_boundary, (N,N+1))
    # push!(x_boundary, (-1,0))
    # push!(x_boundary, (N+1,0))
    # push!(x_boundary, (-1,N))
    # push!(x_boundary, (N+1,N))
    y_boundary=[-1,N+1]
    return x_neighbours, y_neighbours, x_boundary, y_boundary
end



