function losses(r::NeRFRenderer, batch; kwargs...)
    predictions = render_rays(r, batch[:, 1:2, :]; kwargs...)
    targets = batch[:, 3, :]
    coarse_loss = Flux.Losses.mse(predictions.coarse, targets)
    fine_loss = Flux.Losses.mse(predictions.fine, targets)
    total_loss = coarse_loss + fine_loss
    return (total=total_loss, coarse=coarse_loss, fine=fine_loss)
end