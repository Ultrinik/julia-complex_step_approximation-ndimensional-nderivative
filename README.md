# The complex-step derivative approximation
This is a short library that implements the complex-step derivative approximation algorithm for the computation of the N-derivative of an N-dimension function.

This repository also includes the implementation of the Newton Krylov method that uses this method for the jacobian-vector product approximation, using the BifurcationKit library as a base.

## Examples
Inside the folders, you can find dedicated example scripts.

Calculating the directional derivative at a point:
```julia
include("derivative_approximations.jl")

cpoints = [1] # points to use for the complex-step approximation
fpoints = [0,1] # points to use for the finite differences approximation

csa_func, _, _ = get_csa_function(cpoints) # complex-step approximation method
fdf_func, _, _ = get_fdf_function(fpoints) # finite differences approximation method

function func(x, c)
    return sin.(x) .+ cos.(x)
end

x = [1., 2.] # Point of interest
v = [1., 1.] # Direction
h = 1e-8 # Step size

result_csa = csa_func(func, x, v, h)
result_fdf = fdf_func(func, x, v, h)
```

Calculating the 2nd order directional derivative at a point:
Keep in mind that the results may not be very accurate.
```julia
include("derivative_approximations.jl")

cpoints = [1, 2, 3] # points to use for the complex-step approximation
fpoints = [-2,-1,1,2] # points to use for the finite differences approximation

csa_func, _, _ = get_csa_function(cpoints, 2) # complex-step approximation method
fdf_func, _, _ = get_fdf_function(fpoints, 2) # finite differences approximation method

function func(x, c)
    return sin.(x) .+ cos.(x)
end

x = [1., 2., 3.] # Point of interest
v = [1., 1., 1.] # Direction
h = 1e-8 # Step size

result_csa  = csa_func(func, x, v, h)
result_fdf  = fdf_func(func, x, v, h)
```

Calculating the root of a function using the Newton Krylov method:
```julia
include("newton_krylov_methods.jl")

# We ensure that there exists at least one root
function _func(x)
    return sin.(x) .* cos.(x)
end    

xr = [1.0, 2.0, 3.0, 4.0, 5.0]
fr = _func(xr)
n = length(xr)

function func(x, c)
    return _func(x) .- fr
end


x0 = zeros(n) # Inital guess
h = 1e-12 # Step size

solution_csa_method  = newton_krylov_csa(func, x0; rdiff=h)
solution_fdf_method =  newton_krylov_fdf(func, x0; rdiff=h)
```

## Dependencies

* [BifurcationKit](https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/) (Only for the Newton Krylov method)

## References

* Veltz, R., 2020, "BifurcationKit.jl," Inria Sophia-Antipolis, July. Available at: [https://hal.archives-ouvertes.fr/hal-02902346/file/354c9fb0d148262405609eed2cb7927818706f1f.tar.gz](https://hal.archives-ouvertes.fr/hal-02902346/file/354c9fb0d148262405609eed2cb7927818706f1f.tar.gz).
