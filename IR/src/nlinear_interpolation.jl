# Player errors in execution and misjudgement
mistake(rate, value::Real) = value * (1 - 2 * rate) + rate
mistake(rate, value) = @. value * (1 - 2 * rate) + rate

# Player execution of social rules (strategies, norms, etc.)
# Base case(s):
lerp(M::SVector{2}, v::Real) = M' * SA[1 - v, v]
lerp(M::SVector{2}, v::SVector{1}) = lerp(M, v[1])
lerp(M::SMatrix{2,2}, (x, y)::SVector{2}) = SA[1 - x, x]' * M * SA[1 - y, y]
function lerp(M::SArray{NTuple{3,2},T}, (x, y, z)::SVector{3,S}) where {T,S}
    R = promote_type(T, S)
    return lerp(SVector{2,R}(lerp(M[:, :, 1], SA[x, y]), lerp(M[:, :, 2], SA[x, y])), z)
end

# Generic case:
function lerp(M::SArray{NTuple{N,2},T}, v::SVector{N,S}) where {N,T,S}
    SubSA = SArray{NTuple{N - 1,2},T,N - 1,2^(N - 1)}
    R1 = SubSA(selectdim(M, N, 1))
    R2 = SubSA(selectdim(M, N, 2))
    vsub = SVector{N - 1,S}(view(v, 1:(N - 1)))
    return lerp(SA[lerp(R1, vsub), lerp(R2, vsub)], v[N])
end