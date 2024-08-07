using DataStructures

norms = Dict(
    :SimpleStanding => iNorm(243),
    :Shunning => iNorm(192),
    :SternJudging => iNorm(195),
    :ImageScoring => iNorm(240),
)

simple_norms = Dict(
    :SimpleStanding => SA[1 1; 0 1],
    # :Shunning => SA[0 0; 0 1],
    :SternJudging => SA[1 0; 0 1],
    # :ImageScoring => SA[0 0; 1 1],

    :InvSimpleStanding => SA[1 0; 0 0],
    # :InvShunning => SA[1 1; 0 1],
    :SomethingElse => SA[0 1; 1 0],
    # :InvImageScoring => SA[1 1; 0 0]
)

norm_names = OrderedDict([
    evalpoly(2, reshape(value, 4)) => key for (key, value) in simple_norms
])

function get_norm_name!(i::Integer; norm_names=norm_names)
    get!(norm_names, i) do
        Symbol("$i")
    end
end

get_norm_name!.(0:15)
sort!(norm_names)
