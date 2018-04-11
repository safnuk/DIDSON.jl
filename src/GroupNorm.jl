using Flux

struct GroupNorm{F, V, N}
    λ::F  # activation function
    groupsize::Int
    β::V  # bias
    γ::V  # scale
    ϵ::N
end


GroupNorm(dims::Integer...; λ = identity, channels_per_group=8, 
                   initβ = zeros, initγ = ones, ϵ = 1e-8) =
    GroupNorm(λ, channels_per_group, param(initβ(dims)), param(initγ(dims)), ϵ)

function (GN::GroupNorm)(x)
    λ, γ, β, κ = GN.λ, GN.γ, GN.β, GN.groupsize
    T = eltype(x)
    ϵ = Flux.data(convert(T, GN.ϵ))
    dims = size(x)
    C = dims[end-1]
    N = dims[end]
    affine = ones(Int, length(dims))
    affine[end-1] = C
    if C <= κ
        G = 1
    else
        @assert C % κ == 0
        G = div(C, κ)
    end
    x = reshape(x, :, G, N)
    μ = mean(x, 1)
    σ = sqrt.(mean((x .- μ).^2, 1)  .+ ϵ)
    x = reshape(((x .- μ) ./ σ), dims...)
    λ.(reshape(γ, affine...) .* x .+ reshape(β, affine...))
end

Flux.treelike(GroupNorm)
