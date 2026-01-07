# Import required packages
using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, VideoIO
using Revise
using CellularDecisions


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
    x=u1
    y=u2
    z=[(u3[i],s1[i],s2[i],s3[i]) for i in 1:length(u3)]
    #return the coarse grained path
    x_poss=collect(0:N)
    y_poss=collect(0:N)
    z_poss=[]
    # generate z_poss
    s_poss=collect(0:1)
    u3_poss=collect(0:N)
    for s1 in s_poss
        for s2 in s_poss
            for u3 in u3_poss
                for s3 in s_poss
                    push!(z_poss, (u3,s1,s2,s3))
                end
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
    x=u1
    y=u2
    z=(u3,s1,s2,s3)
    return (x,y,z)
end

function coarse_grained_index(x,y,z, cell_system::CellSystem)
    state_dict_inv=cell_system.state_dict_inv
    return state_dict_inv[((x,z[2]),(y,z[3]),(z[1],z[4]))]+1
end

function hidden_var_to_index(z)
    return 8*z[1]+4*z[2]+2*z[3]+z[4]+1
end

function hidden_var_y_to_index(y,z)
    return (8*(N+1))*y+(8*z[1]+4*z[2]+2*z[3]+z[4])+1
end

function hidden_var_x_to_index(x,z)
    return (8*(N+1))*x+(8*z[1]+4*z[2]+2*z[3]+z[4])+1
end


function get_neighbours(cell_system::CellSystem)
    N=cell_system.internal_states
    x_neighbours=[-1,1]
    y_neighbours=[-1,1]
    x_boundary=[-1,N+1]
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





