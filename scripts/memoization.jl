"""
    memoize(foo::Function, n_outputs::Int)

Take a function `foo` and return a vector of length `n_outputs`, where element
`i` is a function that returns the equivalent of `foo(x...)[i]`.

To avoid duplication of work, cache the most-recent evaluations of `foo`.
Because `foo_i` is auto-differentiated with ForwardDiff, our cache needs to
work when `x` is a `Float64` and a `ForwardDiff.Dual`.
"""
function memoize(foo::Function, n_outputs::Int)
    last_x, last_f = nothing, nothing
    last_dx, last_dfdx = nothing, nothing
    function foo_i(i, x::T...) where {T<:Real}
        if T == Float64
            if x != last_x || isnothing(last_f)
                # println("Computing primal $foo at index $i")
                last_x, last_f = x, foo(x...)
            end
            return last_f[i]::T
        else
            if x != last_dx || isnothing(last_dfdx)
                # println("Computing dual $foo at index $i")
                last_dx, last_dfdx = x, foo(x...)
                # @show last_dfdx
            end
            return last_dfdx[i]::T
        end
    end
    @show foo last_x last_f
    return [(x...) -> foo_i(i, x...) for i in 1:n_outputs]
end

