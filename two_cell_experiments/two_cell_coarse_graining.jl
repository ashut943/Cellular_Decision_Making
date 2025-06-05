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
    N=trajectory.internal_states
    u1=u_dict[1]
    u2=u_dict[2]
    s1=s_dict[1]
    s2=s_dict[2]
    #get the coarse grained path
    x=u1
    y=u2
    z=[(s1[i],s2[i]) for i in 1:length(s1)]
    #return the coarse grained path
    x_poss=collect(0:N)
    y_poss=collect(0:N)
    z_poss=[(0,0),(0,1),(1,0),(1,1)]
    #also return the dictionary of the possible states x,y,z
    return x,y,z,x_poss,y_poss,z_poss
end

function get_neighbours(cell_system::CellSystem)
    N=cell_system.internal_states
    x_neighbours=[-1,1]
    y_neighbours=[-1,1]
    x_boundary=[-1,N+1]
    y_boundary=[-1,N+1]
    return x_neighbours, y_neighbours, x_boundary, y_boundary
end

function coarse_grain_tuple(tuple_of_states)
    x=tuple_of_states[1][1]
    y=tuple_of_states[2][1]
    z=(tuple_of_states[1][2],tuple_of_states[2][2])
    return (x,y,z)
end

function coarse_grained_index(x,y,z, cell_system::CellSystem)
    state_dict_inv=cell_system.state_dict_inv
    return state_dict_inv[((x,z[1]),(y,z[2]))]+1
end

function hidden_var_to_index(z)
    return 2*z[1]+z[2]+1
end

function hidden_var_y_to_index(y,z)
    return 4*y+2*z[1]+z[2]+1
end

function hidden_var_x_to_index(x,z)
    return 4*x+2*z[1]+z[2]+1
end
