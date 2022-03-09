@option struct NeRFConfig
    x_dim::Int=3
    d_dim::Int=3
    n_input_layers::Int=5
    n_mid_layers::Int=4
    hidden_dim::Int=256
    color_layer_dim::Int=128
    x_freqs::Int=10
    d_freqs::Int=4
end

struct NeRFModel
    input_layers
    mid_layers
    density
    color
    x_freqs::Int
    d_freqs::Int
end

Flux.@functor NeRFModel (input_layers, mid_layers, density, color,)


function sinusodial_emb(coords, freqs::Int)
    coeffs = 2 .^ range(0, freqs-1)
    inputs = reshape(coords, 1, size(coords)...) .* coeffs
    @check size(inputs) == (freqs, size(coords)...)
    sines = sin.(inputs)
    cosines = cos.(inputs)
    combined = vcat(sines, cosines)
    return reshape(combined, :, size(combined)[begin+2:end]...)
end

function NeRFModel(;config::NeRFConfig)
    x_emb_dim = config.x_dim * 2 * config.x_freqs
    d_emb_dim = config.d_dim * 2 * config.d_freqs
    layers = [Dense(config.hidden_dim, config.hidden_dim, relu) for _ in 2:config.n_input_layers] 
    input_layers = Chain(Dense(x_emb_dim, config.hidden_dim, relu), layers...)

    layers = [Dense(config.hidden_dim, config.hidden_dim, relu) for _ in 2:config.n_mid_layers]
    mid_layers = Chain(Dense(config.hidden_dim+x_emb_dim, config.hidden_dim), layers...)

    density = Dense(config.hidden_dim, 1, softplus)
    color = Chain(Dense(config.hidden_dim+d_emb_dim, config.color_layer_dim, relu), Dense(config.color_layer_dim, 3, tanh_fast))
    return NeRFModel(input_layers, mid_layers, density, color, config.x_freqs, config.d_freqs)
end

function (m::NeRFModel)(x, d)
    x_emb = sinusodial_emb(x, m.x_freqs)
    d_emb = sinusodial_emb(d, m.d_freqs)

    z = m.input_layers(x_emb)
    z1 = vcat(z, x_emb)
    z2 = m.mid_layers(z1)
    density = m.density(z2)
    z3 = vcat(z2, d_emb)
    rgb = m.color(z3)
    return density, rgb
end
