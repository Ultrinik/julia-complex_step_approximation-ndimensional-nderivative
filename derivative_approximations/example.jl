using BenchmarkTools
include("derivative_approximations.jl")

function main()

    cpoints = [1]
    cpoints2 = [1, 2, 3]
    fpoints = [0, 1]
    fpoints2 = [-2, -1, 0, 1, 2]

    csa_func, _, _ = get_csa_function(copy(cpoints), 1, false)
    csa2_func, _, _ = get_csa_function(copy(cpoints2), 1, false)
    fdf_func, _, _ = get_fdf_function(copy(fpoints), 1, false)
    fdf2_func, _, _ = get_fdf_function(copy(fpoints2), 1, false)

    x = [1.0, 2.0]
    v = [1.0, 1.0]
    h = 1e-8

    function func(x, c)
        return sin.(x) .+ cos.(x)
    end

    function derivative_func(x, c)
        return cos.(x) .- sin.(x)
    end

    times = 1000

    rr = derivative_func(x, nothing)
    println("Real result:               ", rr)
    println("")

    r = csa_func(func, x, v, h)
    println("Complex Step error:        ", r - rr, " ", norm(r - rr))
    t = @benchmark $csa_func($func, $x, $v, $h) samples = times
    execution_time = mean(t).time / 1e9
    println("Execution Time:            ", execution_time/times)

    r = csa2_func(func, x, v, h)
    println("Complex Step 2 error:      ", r - rr, " ", norm(r - rr))
    t = @benchmark $csa2_func($func, $x, $v, $h) samples = times
    execution_time = mean(t).time / 1e9
    println("Execution Time:            ", execution_time/times)

    println("")

    r = fdf_func(func, x, v, h)
    println("Finite Difference error:   ", r - rr, " ", norm(r - rr))
    t = @benchmark $fdf_func($func, $x, $v, $h) samples = times
    execution_time = mean(t).time / 1e9
    println("Execution Time:            ", execution_time/times)

    r = fdf2_func(func, x, v, h)
    println("Finite Difference 2 error: ", r - rr, " ", norm(r - rr))
    t = @benchmark $fdf2_func($func, $x, $v, $h) samples = times
    execution_time = mean(t).time / 1e9
    println("Execution Time:            ", execution_time/times)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end