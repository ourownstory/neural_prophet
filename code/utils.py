import torch


def intelligent_regularization(sparsity):
    if sparsity is not None:
        lam = 0.02 * (1.0 / sparsity - 1.0)
    else:
        lam = 0.0
    return lam


def train_batch(model, x, y, optimizer, loss_fn, lambda_value=None):
    # Run forward calculation
    y_predict = model.forward(x)

    # Compute loss.
    loss = loss_fn(y_predict, y)

    # regularize
    # Warning: will grab first layer as the weight to be regularized!
    if lambda_value is not None:
        reg_loss = torch.zeros(1, dtype=torch.float, requires_grad=True)
        abs_weights = torch.abs(model.layers[0].weight)
        reg = torch.div(2.0, 1.0 + torch.exp(-3.0*abs_weights.pow(1.0/3.0))) - 1.0
        reg_loss = reg_loss + torch.mean(reg)

        loss = loss + lambda_value * reg_loss

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    return loss.data.item()
