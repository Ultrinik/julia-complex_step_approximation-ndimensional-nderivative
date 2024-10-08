using BenchmarkTools
include("newton_krylov_methods.jl")

function main()

    function _func(x)
        return x.^2 .- sin.(x) .* cos.(x)
    end    
    
    xr = [1.0, 2.0, 3.0, 4.0, 5.0]
    n = length(xr)

    function func(x, c)
        return _func(x) .- _func(xr)
    end
    
    x0 = zeros(n)
    newton_iters = 100
    inner_maxiter = 20
    tol = 0.0
    h = 1e-12

    times = 100
    
    println("Original Solution:                   ", xr)
    
    println("")

    solution = newton_krylov(func, x0; iters=newton_iters, tol=tol)
    println("Default BifurcationKit Solution:     ", solution, " Error: ", norm(func(solution, nothing)))
    t = @benchmark $newton_krylov($func, $x0; iters=$newton_iters, tol=$tol) samples = times
    execution_time = mean(t).time / 1e9
    println("Execution Time:                      ", execution_time/times)

    println("")

    solution = newton_krylov_fdf(func, x0; iters=newton_iters, inner_maxiter=inner_maxiter, tol=tol, rdiff=h)
    println("Finite Difference Solution:          ", solution, " Error: ", norm(func(solution, nothing)))
    t = @benchmark $newton_krylov_fdf($func, $x0; iters=$newton_iters, inner_maxiter=$inner_maxiter, tol=$tol, rdiff=$h) samples = times
    execution_time = mean(t).time / 1e9
    println("Execution Time:                      ", execution_time/times)
    
    println("")

    solution = newton_krylov_csa(func, x0; iters=newton_iters, inner_maxiter=inner_maxiter, tol=tol, rdiff=h)
    println("Complex Step Solution:               ", solution, " Error: ", norm(func(solution, nothing)))
    t = @benchmark $newton_krylov_csa($func, $x0; iters=$newton_iters, inner_maxiter=$inner_maxiter, tol=$tol, rdiff=$h) samples = times
    execution_time = mean(t).time / 1e9
    println("Execution Time:                      ", execution_time/times)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end