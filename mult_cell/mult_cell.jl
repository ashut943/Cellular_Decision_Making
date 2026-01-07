# multiple cell module, for definition of the problem and optimization functions (using interior point method)

using JuMP, Ipopt, LinearAlgebra, Distributions, DataStructures, Revise
using CellularDecisions
using AppleAccelerate

include("src/mult_cell_nonlinear.jl")
include("src/mult_cell_setup.jl")
include("src/mult_cell_hittingtime.jl")
include("src/mult_cell_hittingprob.jl")
