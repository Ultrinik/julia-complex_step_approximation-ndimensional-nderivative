using LinearAlgebra, Printf

"""
    get_csa_function(points::Vector, derivate::Int = 1, print_mode::Bool = false, dtype::Type = Complex{Float64})

Constructs a Complex Step Approximation function using the given points.

# Arguments
- `points::Vector`: A vector of points to use for the approximation.
- `derivate::Int`: The order of the derivative to calculate. Default is `1`.
- `print_mode::Bool`: Flag to enable debug prints (set to `true` for debugging). Default is `false`.
- `dtype::Type`: The complex number data type to use, typically either `Complex{Float64}` or `Complex{Float32}`. Default is `Complex{Float64}`.

# Returns
- A function that takes the arguments `(func, x, v, h)`.
- A tuple containing the convergence order of the approximation.
- A vector of coefficients used in the approximation, in the same order as the input `points` vector.
"""
function get_csa_function(points::Vector, derivate::Int = 1, print_mode::Bool = false, dtype::Type = Complex{Float64})

    # Remove point 0 if derivative is odd, it will always be 0
    if derivate % 2 == 1 && 0 in points
        deleteat!(points, findfirst(==(0), points))
    end

    # Decide if method needs real or imaginary component
    component_func = derivate % 2 == 1 ? imag : real

    # Get coefficients of Taylor Expansion
    function get_coefficients(points)
        coefficients = []
        for point in points
            current_coefficients = zeros(dtype, 2 * length(points))
            for i in 0:(2 * length(points) - 1)
                power = (1im * dtype(point))^i
                coefficient = power / factorial(i)
                current_coefficients[i + 1] = coefficient
            end
            push!(coefficients, current_coefficients) #here each push shoudl be a new row of the matrix
        end
        coefficients_matrix = hcat(coefficients...)
        return component_func(coefficients_matrix)
    end

    main_coefficients = get_coefficients(points)

    # Get coefficients of Taylor Expansion with one more point
    push!(points, maximum(abs.(points)) + 1)
    push!(points, maximum(abs.(points)) + 1)
    extended_coefficients_og = get_coefficients(points)
    extended_non_zero_rows = findall(row -> all(row .== 0.0), eachrow(extended_coefficients_og))
    extended_zero_row_indices = collect(extended_non_zero_rows)
    extended_coefficients_filtered = extended_coefficients_og[setdiff(1:size(extended_coefficients_og, 1), extended_zero_row_indices), :]

    # Solve system
    A = main_coefficients
    b = zeros(eltype(A), size(A,1))
    b[derivate + 1] = 1

    # Identify rows that are all zeros
    zero_rows = findall(row -> all(row .== 0.0), eachrow(A))
    zero_row_indices = collect(zero_rows)

    # Filter out rows that contain only zeros
    A_filtered = A[setdiff(1:size(A, 1), zero_row_indices), :]
    b_filtered = b[setdiff(1:size(b, 1), zero_row_indices)]

    if print_mode
        println("Matrix A_filtered:")
        println(A_filtered)
        println("Array b_filtered:")
        println(b_filtered)
    end

    solution = nothing
    for i in 1:5
        try
            solution = A_filtered \ b_filtered
            break
        catch e
            if isa(e, LinearAlgebra.LinAlgError)
                A_filtered = A_filtered[1:end-1, 1:end-1]
                b_filtered = b_filtered[1:end-1]
    
                if print_mode
                    println("Error: A complex-step method cannot be generated with the chosen points.")
                    if i < 5
                        println("Adjusting Points")
                    end
                    
                    println("($(i)) Matrix A:")
                    println(A_filtered)
                    println("($(i)) Array b:")
                    println(b_filtered)
                end
            else
                println("An unexpected error occurred: ", e)
                return nothing, nothing, nothing
            end
        end
    end

    # Convergence
    extended_coefficients_filtered_cutted = extended_coefficients_filtered[:, 1:length(solution)]
    extended_b = extended_coefficients_filtered_cutted * solution

    extended_b_non_zero_values = extended_b[extended_b .!= 0]
    second_non_zero_value = extended_b_non_zero_values[1]
    second_non_zero_index = findfirst(extended_b .== second_non_zero_value)

    o_factor = second_non_zero_value
    o_factor_exp = 2 * second_non_zero_index

    final_solution = copy(solution)

    if print_mode
        println("Solution X:")
        println(final_solution)
        println("A @ Solution:")
        println(A_filtered * final_solution)
        println("ext_A @ Solution:")
        println(extended_b)
        println("Order:")
        @printf("O(%.5f h^%d)\n", abs(o_factor), o_factor_exp)
    end

    # Function
    """
        csa_function(F::Function, x::Union{Float64, AbstractArray}, v::Union{Float64, AbstractArray}, h::Float64 = 1e-8)

    Computes the Complex Step approximation of the function F at a point x.

    # Parameters
    - `F::Function`: A callable function that takes a single input (either a `Float` or an `Array`).
    - `x::Union{Float, AbstractArray}`: A `Float` or `AbstractArray` representing the point(s) at which to evaluate the approximation.
    - `v::Union{Float, AbstractArray}`: A `Float` or `AbstractArray` representing the perturbation values for the approximation, which must be of the same type as `x`.
    - `h::Float`: The step size for the approximation. Default is `1e-8`.

    # Returns
    - A `Float` or `AbstractArray` representing the approximated value(s) of the function F at the specified point(s) x.
    """
    function csa_function(F::Function, x::Union{Float64, AbstractArray}, v::Union{Float64, AbstractArray}, h::Float64 = 1e-8)
        total = zeros(dtype, size(x)...)

        for i in 1:(length(final_solution))
            point = points[i]
            final_c = final_solution[i]

            step = v * h * 1im
            arguments = convert(Vector{dtype}, x .+ step * point)
            
            total += final_c * F(arguments, nothing)
        end
        return component_func(total) / h^derivate
    end

    return csa_function, (o_factor, o_factor_exp), final_solution
end

"""
    get_fdf_function(points::Vector, derivate::Int = 1, print_mode::Bool = false, dtype::Type = Float64)

Constructs a Finite Difference Approximation function using the given points.

# Arguments
- `points::Vector`: A vector of points to use for the approximation.
- `derivate::Int`: The order of the derivative to calculate. Default is `1`.
- `print_mode::Bool`: Flag to enable debug prints (set to `true` for debugging). Default is `false`.
- `dtype::Type`: The data type to use, typically `Float64`. Default is `Float64`.

# Returns
- A function that takes the arguments `(func, x, v, h)`.
- A tuple containing the convergence order of the approximation.
- A vector of coefficients used in the approximation, in the same order as the input `points` vector.
"""
function get_fdf_function(points::Vector, derivate::Int = 1, print_mode::Bool = false, dtype::Type = Float64)

    # Get coefficients of Taylor Expansion
    function get_coefficients(points)
        coefficients = []
        for point in points
            current_coefficients = zeros(dtype, length(points))
            for i in 0:(length(points) - 1)
                power = point^i
                coefficient = dtype(power / factorial(i))
                current_coefficients[i + 1] = coefficient
            end
            push!(coefficients, current_coefficients)
        end
        coefficients_matrix = hcat(coefficients...)
        return coefficients_matrix
    end

    main_coefficients = get_coefficients(points)

    # Get coefficients of Taylor Expansion with one more point
    push!(points, maximum(abs.(points)) + 1)
    extended_coefficients_og = get_coefficients(points)
    extended_coefficients = extended_coefficients_og[:, 1:end-1]

    # Solve system
    A = main_coefficients
    b = zeros(eltype(A), size(A,1))
    b[derivate + 1] = 1
    solution = A \ b

    # Convergence
    extended_b = extended_coefficients * solution
    o_factor = extended_b[end]
    o_factor_exp = length(extended_b) - 2

    final_solution = copy(solution)

    if print_mode
        println("Matrix A:")
        println(A)
        println("Array b:")
        println(b)
        println("Solution X:")
        println(final_solution)
        println("A @ Solution:")
        println(A * final_solution)
        println("ext_A @ Solution:")
        println(extended_b)
        println("Order:")
        @printf("O(%.5f h^%d)\n", abs(o_factor), o_factor_exp)
    end

    # Function
    """
        fdf_function(F::Function, x::Union{Float64, AbstractArray}, v::Union{Float64, AbstractArray}, h::Float64 = 1e-8)

    Computes the Finite Difference approximation of the function F at a point x.

    # Parameters
    - `F::Function`: A callable function that takes a single input (either a `Float` or an `Array`).
    - `x::Union{Float, AbstractArray}`: A `Float` or `AbstractArray` representing the point(s) at which to evaluate the approximation.
    - `v::Union{Float, AbstractArray}`: A `Float` or `AbstractArray` representing the perturbation values for the approximation, which must be of the same type as `x`.
    - `h::Float`: The step size for the approximation. Default is `1e-8`.

    # Returns
    - A `Float` or `AbstractArray` representing the approximated value(s) of the function F at the specified point(s) x.
    """
    function fdf_function(F::Function, x::Union{Float64, AbstractArray}, v::Union{Float64, AbstractArray}, h::Float64 = 1e-8)
        total = zero(x)
        for i in 1:(length(points) - 1)
            point = points[i]
            final_c = final_solution[i]
            step = v * h
            total += final_c * F(x .+ step * point, nothing)
        end
        return total / h^derivate
    end

    return fdf_function, (o_factor, o_factor_exp), final_solution
end