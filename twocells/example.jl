using ForwardDiff, LinearAlgebra, Optim

# Define your A(P) and b(P) functions (customize as needed)
# Define A(P) = (1 + p₁² + p₂²) * I, ensuring A(P) is always invertible.
function A_of(P)
    p1, p2 = P
    return (1 + p1^2 + p2^2) * I(2)
end

# Define b(P) = [p₁ - 1, p₂ - 2]
function b_of(P)
    p1, p2 = P
    return [p1 - 1, p2 - 2]
end

# Define the objective function:
# Solve A(P) * τ = b(P) and return τ[1]

# Objective function: solves A(P)τ = b(P) and returns τ[1]
function objective(P)
    A = A_of(P)
    b = b_of(P)
    τ = A \ b
    return τ[1]
end

# Custom gradient descent function
function gradient_descent(objective, P0; lr=0.01, tol=1e-6, max_iter=1000)
    P = copy(P0)
    for iter in 1:max_iter
        grad = ForwardDiff.gradient(objective, P)
        # Update parameters
        P_new = P - lr * grad
        # Check convergence (using the change in parameters)
        if norm(P_new - P) < tol
            println("Converged in iteration ", iter)
            return P_new
        end
        P = P_new
    end
    println("Reached maximum iterations.")
    return P
end

# Initial guess for parameters P
P0 = [1.0, 1.0]
P_opt = gradient_descent(objective, P0; lr=0.05, tol=1e-6, max_iter=1000)

println("Optimized parameters: ", P_opt)
println("Objective value at optimum: ", objective(P_opt))
#!/usr/bin/env julia


# Define the objective function that computes τ from the linear system A(P)*τ = b(P)
# and returns the first element τ[1].
function objective(P)
    A = A_of(P)
    b = b_of(P)
    τ = A \ b  # Solve the linear system
    return τ[1]
end

# Optionally, define a function to compute the gradient at a given P.
function compute_gradient(P)
    return ForwardDiff.gradient(objective, P)
end

# Main script execution
function main()
    # Initial guess for the parameter vector P
    P0 = [1.0, 1.0]
    
    # Compute and display the gradient at the initial point
    grad = compute_gradient(P0)
    println("Gradient at P0: ", grad)
    
    # Create an options instance with the maximum number of iterations
    opts = Optim.Options(iterations = 1000)
    
    # Optimize the objective using gradient descent
    result = optimize(objective, P0, GradientDescent(), opts)
    
    # Display the optimized parameters and the minimum objective value
    println("Optimized parameters: ", Optim.minimizer(result))
    println("Minimum value: ", Optim.minimum(result))
end

# Run the main function when the script is executed
main()
# Predefined parameter lambda (this is not optimized)
const λ = 1.0

# Define A(P, λ) and b(P, λ) where lambda is fixed.
function A_of(P, λ)
    p1, p2 = P
    # For instance, let A(P, λ) be a matrix that uses λ
    return (λ + p1^2 + p2^2) * I(2)
end

function b_of(P, λ)
    p1, p2 = P
    # Define b(P, λ) that uses λ as well
    return [p1 - 1, p2 - 2]
end

# Define the objective function that solves A(P, λ)*τ = b(P, λ)
# and returns the first element of τ. Note that lambda is passed in but fixed.
function objective(P, λ)
    A = A_of(P, λ)
    b = b_of(P, λ)
    τ = A \ b  # Solve A(P, λ)*τ = b(P, λ)
    return τ[1]
end

function main()
    # Initial guess for the parameter vector P
    P0 = [1.0, 1.0]
    
    # Define an objective function where lambda is fixed.
    fixed_objective = P -> objective(P, λ)
    
    # Compute and display the gradient at the initial point using the fixed objective.
    grad = ForwardDiff.gradient(fixed_objective, P0)
    println("Gradient at P0: ", grad)
    
    # Set up optimization options
    opts = Optim.Options(iterations = 1000)
    
    # Optimize using gradient descent on the fixed objective
    result = optimize(fixed_objective, P0, GradientDescent(), opts)
    
    println("Optimized parameters: ", Optim.minimizer(result))
    println("Minimum value: ", Optim.minimum(result))
end

# Run the main function
main()