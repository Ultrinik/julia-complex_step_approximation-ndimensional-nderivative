using Printf, FileIO
using LinearAlgebra
using Random
using Printf
using Distributions
using DelimitedFiles

include("experimento.jl")
include("exp_base.jl")

n = 5
seed = 32

A = obtain_A_matrix("uniform", n)
#println(A)
obtain_sin_function_ = (n, s) -> obtain_sin_function(n, A, s)
obtain_uniform_function_ = (n, s) -> obtain_uniform_function(n, A, s)
obtain_euler_function_ = (n, s) -> obtain_euler_function(n, A, s)

A = obtain_A_matrix("normal", n)
#println(A)
obtain_normal_function_ = (n, s) -> obtain_normal_function(n, A, s)

funciones = [obtain_sin_function_, obtain_normal_function_, obtain_uniform_function_,
             obtain_sparse1_function, obtain_sparse2_function, obtain_dense1_function,
             obtain_dense2_function, obtain_euler_function_]
funciones_string = ["obtain_sin_function", "obtain_normal_function", "obtain_uniform_function",
            "obtain_sparse1_function", "obtain_sparse2_function", "obtain_dense1_function",
            "obtain_dense2_function", "obtain_euler_function"]



if false
    for (func, func_str) in zip(funciones, funciones_string)
        println("Function: ", func_str)
        for i in 1:1
            result = jacobian_free_newton_krylov(func, n)
            println("Result ", i, ": ", result)
        end
        println("")
    end
else

    xvec = Float64[0,0,0,0,0]
    F, x0f = obtain_sparse1_function(n,seed)
    println(F(xvec, nothing))
    println(norm(F(xvec, nothing)))
    println("")

    result = jacobian_free_newton_krylov(obtain_sparse1_function, n, total_iter=[1,0])
    println(result)
    println(F(result, nothing))
    println(norm(F(result, nothing)))
end
