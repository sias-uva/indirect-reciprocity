# Player errors in execution and misjudgement
"""
    mistake(rate, value)

Return the expected value of the random variable X where:
- ℙ(X = `value`) = 1 - rate
- ℙ(X = 1 - `value`) = rate

This is performed elementwise for non-atomic `rate` and `value`.
"""
mistake(rate, value) = @. value * (1 - 2 * rate) + rate
mistake(rate, value::Real) = value * (1 - 2 * rate) + rate

"""
    execution_oopsie(rate, value::Real)

- `value`: intended probability of cooperation
- `rate`: rate at which cooperation accidentally fails

NB: defection cannot accidentally fail
"""
function execution_oopsie(rate, value::Real)
    # ℙ(C) = 1 - ℙ(D)
    # ℙ(D) = (1 - value) + (rate * value)
    return (1 - rate) * value
end

# Player execution of social rules (strategies, norms, etc.)
# Base case(s):
lerp(M::SVector{2}, v::Real) = M' * SA[1 - v, v]
lerp(M::SVector{2}, v::SVector{1}) = lerp(M, v[1])
lerp(M::SMatrix{2,2}, (x, y)::SVector{2}) = SA[1 - x, x]' * M * SA[1 - y, y]
function lerp(M::SArray{NTuple{3,2},T}, (x, y, z)::SVector{3,S}) where {T,S}
    R = promote_type(T, S)
    return lerp(SVector{2,R}(lerp(M[:, :, 1], SA[x, y]), lerp(M[:, :, 2], SA[x, y])), z)
end

# Generic case (should never be called as we have specialised methods for up to 3 dims):
"""
    lerp(M::SArray{NTuple{N,2},T}, v::SVector{N,S}) where {N,T,S}

N-linear interpolation of `M` at the value `v`. `v` is expected to have the same
length as the number of dimensions as `M` and its values should be in the
interval [0, 1].
"""
function lerp(M::SArray{NTuple{N,2},T}, v::SVector{N,S}) where {N,T,S}
    SubSA = SArray{NTuple{N - 1,2},T,N - 1,2^(N - 1)}
    R1 = SubSA(selectdim(M, N, 1))
    R2 = SubSA(selectdim(M, N, 2))
    vsub = SVector{N - 1,S}(view(v, 1:(N - 1)))
    return lerp(SA[lerp(R1, vsub), lerp(R2, vsub)], v[N])
end
