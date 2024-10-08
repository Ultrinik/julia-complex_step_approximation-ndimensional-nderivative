using LinearAlgebra
using BifurcationKit

# The 'method' argument is unused

"""
    newton_krylov_csa(F::Function, x0::Union{Float64, AbstractArray}; 
                      method::String = "gmres", 
                      iters::Int = 50, 
                      inner_maxiter::Int = 20, 
                      callback::Union{Function, Nothing} = nothing, 
                      tol::Float64 = 0.0, 
                      rdiff::Float64 = sqrt(eps(Float64)), 
                      data_type::Type = Complex{Float64})

Finds the root of the function F using the Newton-Krylov method with Complex Step Approximation.

# Parameters
- `F::Function`: A callable function that takes a single input (either a `Float` or an `AbstractArray`).
- `x0`: The initial guess for the root (a `Float` or an `AbstractArray`).
- `method::String`: The iterative method to use for solving the linear systems, default is `"gmres"`.
- `iters::Int`: The maximum number of iterations for the Newton-Krylov method. Default is `50`.
- `inner_maxiter::Int`: The maximum number of inner iterations for the linear solver. Default is `20`.
- `callback::Function`: An optional function to call at each iteration for logging or monitoring progress. Default is `nothing`.
- `tol::Float64`: The convergence tolerance. Default is `0.0` (no tolerance).
- `rdiff::Float64`: The relative difference tolerance. Default is the square root of machine epsilon for `Float64`.
- `data_type::Type`: The data type for the complex numbers used in the computations, default is `Complex{Float64}`.

# Returns
- A tuple containing the estimated root and the number of iterations used.
"""
function newton_krylov_csa(F::Function, x0::Union{Float64, AbstractArray}; 
                            method::String = "gmres", 
                            iters::Int = 50, 
                            inner_maxiter::Int = 20, 
                            callback::Union{Function, Nothing} = nothing, 
                            tol::Float64 = 0.0, 
                            rdiff::Float64 = sqrt(eps(Float64)), 
                            data_type::Type = Complex{Float64})

    function J_finite_csa(x, rdiff_vec)
        Jv = function (v)
    
            nv = norm(v)
            if nv == 0
                return 0 * v
            end
            h = rdiff_vec[2]
            sc = h / nv
            
            args = convert(Vector{data_type}, x .+ 1im * sc * v)
            dr = imag(F(args, nothing)) ./sc
            
            if !all(isfinite, dr) && all(isfinite, v)
                throw(ArgumentError("Function returned non-finite results"))
            end
            return dr
        end
        return Jv
    end
    
    function inner_callback(state; kwargs...)
        residual = state.residual
        if callback !== nothing
            callback(residual)
        end

        return true
    end
    
    rdiff_vec = [rdiff,rdiff]
    prob = BifurcationProblem(F, x0, rdiff_vec; J=J_finite_csa)
    par = NewtonPar(max_iterations=iters, tol=tol, verbose=false, linsolver=GMRESIterativeSolvers(maxiter=inner_maxiter))
    
    solution = newton(prob, par; callback=inner_callback, rdiff_vec=rdiff_vec)
    return solution.u
end

"""
    newton_krylov_fdf(F::Function, x0::Union{Float64, AbstractArray}; 
                      method::String = "gmres", 
                      iters::Int = 50, 
                      inner_maxiter::Int = 20, 
                      callback::Union{Function, Nothing} = nothing, 
                      tol::Float64 = 0.0, 
                      rdiff::Float64 = sqrt(eps(Float64)))

Finds the root of the function F using the Newton-Krylov method with Finite Difference Approximation.

# Parameters
- `F::Function`: A callable function that takes a single input (either a `Float` or an `AbstractArray`).
- `x0`: The initial guess for the root (a `Float` or an `AbstractArray`).
- `method::String`: The iterative method to use for solving the linear systems, default is `"gmres"`.
- `iters::Int`: The maximum number of iterations for the Newton-Krylov method. Default is `50`.
- `inner_maxiter::Int`: The maximum number of inner iterations for the linear solver. Default is `20`.
- `callback::Function`: An optional function to call at each iteration for logging or monitoring progress. Default is `nothing`.
- `tol::Float64`: The convergence tolerance. Default is `0.0` (no tolerance).
- `rdiff::Float64`: The relative difference tolerance. Default is the square root of machine epsilon for `Float64`.

# Returns
- A tuple containing the estimated root and the number of iterations used.
"""
function newton_krylov_fdf(F::Function, x0::Union{Float64, AbstractArray}; 
                            method::String = "gmres", 
                            iters::Int = 50, 
                            inner_maxiter::Int = 20, 
                            callback::Union{Function, Nothing} = nothing, 
                            tol::Float64 = 0.0, 
                            rdiff::Float64 = sqrt(eps(Float64)))

    function J_finite_diff(x, rdiff_vec)
        Jv = function (v)
    
            nv = norm(v)
            if nv == 0
                return 0 * v
            end
            h = rdiff_vec[2]
            sc = h / nv
    
            dr = (F(x .+ sc * v, nothing) .- F(x, nothing)) ./sc
            
            if !all(isfinite, dr) && all(isfinite, v)
                throw(ArgumentError("Function returned non-finite results"))
            end
    
            return dr
        end
        return Jv
    end
    
    function inner_callback(state; kwargs...)
        residual = state.residual
        if callback !== nothing
            callback(residual)
        end

        rdiff_vec = Base.get(kwargs, :rdiff_vec, nothing)
        mx = maximum(abs.(state.x))
        mf = maximum(abs.(state.fx))
        rdiff_vec[2] = rdiff_vec[1] * max(1, mx) / max(1, mf)
        return true
    end
    
    rdiff_vec = [rdiff,rdiff]
    prob = BifurcationProblem(F, x0, rdiff_vec; J=J_finite_diff)
    par = NewtonPar(max_iterations=iters, tol=tol, verbose=false, linsolver=GMRESIterativeSolvers(maxiter=inner_maxiter))
    
    solution = newton(prob, par; callback=inner_callback, rdiff_vec=rdiff_vec)
    return solution.u
end

"""
    newton_krylov(F::Function, x0::Union{Float64, AbstractArray}; 
                  method::String = "gmres", 
                  iters::Int = 50, 
                  callback::Union{Function, Nothing} = nothing, 
                  tol::Float64 = 0.0)

Finds the root of the function F using the Newton-Krylov method.

# Parameters
- `F::Function`: A callable function that takes a single input (either a `Float` or an `AbstractArray`).
- `x0`: The initial guess for the root (a `Float` or an `AbstractArray`).
- `method::String`: The iterative method to use for solving the linear systems, default is `"gmres"`.
- `iters::Int`: The maximum number of iterations for the Newton-Krylov method. Default is `50`.
- `callback::Function`: An optional function to call at each iteration for logging or monitoring progress. Default is `nothing`.
- `tol::Float64`: The convergence tolerance. Default is `0.0` (no tolerance).

# Returns
- A tuple containing the estimated root and the number of iterations used.
"""
function newton_krylov(F::Function, x0::Union{Float64, AbstractArray}; 
                       method::String = "gmres", 
                       iters::Int = 50, 
                       callback::Union{Function, Nothing} = nothing, 
                       tol::Float64 = 0.0)

    function inner_callback(state; kwargs...)
        residual = state.residual
        if callback !== nothing
            callback(residual)
        end

        return true
    end

    prob = BifurcationProblem(F, x0, nothing)
    par = NewtonPar(max_iterations=iters, tol=tol, verbose=false)
    
    solution = newton(prob, par; callback=inner_callback)
    return solution.u
end
