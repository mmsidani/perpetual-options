module MCOpPerp

using Random
using StatsBase
using Distributions
using DataFrames
using CSV

@enum Opttype put=1 call=2

function uniforms(n)
    rand(Random.MersenneTwister(), n)
end


function exponential_inverse_transform(unis, funding_period)
    [funding_period * log(1.0 / (1 - x)) for x in unis]
end


function geometric_variates(n, funding_frequency)
    p = 1 / (funding_frequency + 1)
    # Geometric() returns number of failures before success. we add 1 to get number of trials till first success
    rand(Random.MersenneTwister(), Distributions.Geometric(p), n) .+ 1
end


function standard_error(v, n)
    StatsBase.std(v) / sqrt(n)
end


function black_scholes(t, k, s, sig, opttype)
    sigt = sig * sqrt(t)
    d1 = log(s / k) / sigt + sigt / 2.0
    d2 = d1 - sigt

    if opttype == call
        Distributions.cdf(Normal(), d1) * s - Distributions.cdf(Normal(), d2) * k
    else
        Distributions.cdf(Normal(), -d2) * k - Distributions.cdf(Normal(), -d1) * s
    end
end


# the exact formula from the deri protocol
function perp_exact_deri(k, s, sig, opttype, funding_period)
    u = sqrt(1 + 8 / (sig ^ 2 * funding_period))
    v = k / u * (s >= k ?  (s / k) ^ ((1 - u) / 2) : (s / k) ^ ((u + 1) / 2))
    if opttype == call
        max(s - k, 0) + v
    else
        max(k - s, 0) + v
    end
end


function monte_carlo_continuous(k, n, s, sig, opttype, funding_period)
    unis = uniforms(n)
    expies = exponential_inverse_transform(unis, funding_period)

    v = black_scholes.(expies, k, s, sig, opttype)

    sum(v) / n, StatsBase.std(v) / sqrt(n)
end


function monte_carlo_discrete(k, n, s, sig, opttype, funding_period, funding_frequency)
    expies = geometric_variates(n, funding_frequency) .* funding_period ./ funding_frequency
    v = black_scholes.(expies, k, s, sig, opttype)

    sum(v) / n, StatsBase.std(v) / sqrt(n)
end


function test(filename=nothing; is_continuous=true)
    ns = [4000000, 1000000, 250000]
    s = 106000
    ks = s .- [-150, -100, -50, 0, 50, 100, 150] * 10
    sig = 0.004 # per sqrt(hour) which is about 50% annual
    funding_period = 8 # hours. must be consistent with sig
    funding_frequency = 8
    opttype = call
    ret = DataFrame(strike = ks)
    if is_continuous
        perps = perp_exact_deri.(ks, s, sig, opttype, funding_period)
        perps = round.(perps, digits=2)
        ret = hcat(ret, DataFrame(exact=perps))
    end
    for n in ns
        if is_continuous
            mcres = monte_carlo_continuous.(ks, n, s, sig, opttype, funding_period)
        else
            mcres = monte_carlo_discrete.(ks, n, s, sig, opttype, funding_period, funding_frequency)
        end
        mcs = first.(mcres)
        stdes = last.(mcres)
        label = n >= 1000000 ? string( n / 1000000 |> Int) * "m" : string(n / 1000 |> Int) * "k"
        ret = hcat(ret, DataFrame(label => round.(mcs, digits=2), label * "se" => round.(stdes, digits=4)))
        if is_continuous
            ret = hcat(ret, DataFrame(label * "err" => round.(abs.(perps.-mcs) ./ perps, digits=4)))
        end
    end
        
    ret = sort(ret, :strike)
    if !isnothing(filename)
        ret |> CSV.write(filename)
    end

    ret
end


end # module