import time

import numpy as np
import torch
import torch.nn as nn
from model import DAR
from torch import optim
from torch.utils.data import DataLoader

from v0_1_pure_pytorch import utils


# Architecture, batching etc of DARMA
def train_batch(model, x, y, optimizer, loss_fn, lambda_value=None):
    # Run forward calculation
    y_predict = model.forward(x)

    # Compute loss.
    loss = loss_fn(y_predict, y)

    # regularize
    if lambda_value is not None:
        reg_loss = torch.zeros(1, dtype=torch.float, requires_grad=True)
        if model.num_layers == 1:
            abs_weights = torch.abs(model.layer_1.weight)
            # classic L1
            # reg = torch.mean(abs_weights)

            # sqrt - helps to protect some weights and bring others to zero,
            # but is still tough on larger weights
            # reg = torch.mean(torch.sqrt(abs_weights))

            # new, less hard on larger weights: (protects ~0.1-1.0)
            # reg = torch.div(2.0, 1.0 + torch.exp(-5.0*abs_weights.pow(0.4))) - 1.0

            # mid-way, more stable
            reg = torch.div(2.0, 1.0 + torch.exp(-3.0*abs_weights.pow(1.0/3.0))) - 1.0

            reg_loss = reg_loss + torch.mean(reg)
        else:
            # for weights in model.parameters():
            raise NotImplementedError("L1 Norm for deeper models not implemented")

        loss = loss + lambda_value * reg_loss

    optimizer.zero_grad()

    loss.backward(retain_graph=True)

    optimizer.step()

    return loss.data.item()


def train(model, loader, loss_fn, lr, epochs, lr_decay, est_sparsity, lambda_delay=None, verbose=False):

    # Initialize the optimizer with above parameters
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)

    losses = list()
    batch_index = 0
    epoch_losses = []
    avg_losses = []
    lambda_value = utils.intelligent_regularization(est_sparsity)

    for e in range(epochs):
        # slowly increase regularization until lambda_delay epoch
        if lambda_delay is not None and e < lambda_delay:
            l_factor = e / (1.0 * lambda_delay)
            # l_factor = (e / (1.0 * lambda_delay))**2
        else:
            l_factor = 1.0

        for x, y in loader:
            loss = train_batch(model=model, x=x, y=y, optimizer=optimizer,
                               loss_fn=loss_fn, lambda_value=l_factor*lambda_value)
            epoch_losses.append(loss)
            batch_index += 1
        scheduler.step()
        losses.extend(epoch_losses)
        avg_loss = np.mean(epoch_losses)
        avg_losses.append(avg_loss)
        epoch_losses = []
        if verbose:
            print("{}. Epoch Avg Loss: {:10.2f}".format(e + 1, avg_loss))
    if verbose:
        print("Total Batches: ", batch_index)

    return losses, avg_losses


def test_batch(model, x, y, loss_fn):
    # run forward calculation
    y_predict = model.forward(x)
    loss = loss_fn(y_predict, y)

    return y_predict, loss


def test(model, loader, loss_fn):
    losses = list()
    y_vectors = list()
    y_predict_vectors = list()

    batch_index = 0
    for x, y in loader:
        y_predict, loss = test_batch(model=model, x=x, y=y, loss_fn=loss_fn)

        losses.append(loss.data.numpy())
        y_vectors.append(y.data.numpy())
        y_predict_vectors.append(y_predict.data.numpy())

        batch_index += 1

    losses = np.array(losses)
    y_predict_vector = np.concatenate(y_predict_vectors)
    mse = np.mean((y_predict_vector - np.concatenate(y_vectors)) ** 2)

    return y_predict_vector, losses, mse


def run_train_test(dataset_train, dataset_test, model_config, train_config, verbose=False):
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=train_config["batch"], shuffle=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=len(dataset_test), shuffle=False)

    if model_config["ma"] > 0:
        # TODO: implement DARMA
        raise NotImplementedError
    else:
        del model_config["ma"]
        model = DAR(
            **model_config
        )

    # Define the loss function
    loss_fn = nn.MSELoss()  # mean squared error

    # Train and get the resulting loss per iteration
    del train_config["batch"]
    losses, avg_losses = train(
        model=model,
        loader=data_loader_train,
        loss_fn=loss_fn,
        **train_config,
        verbose=verbose,
    )

    # Test and get the resulting predicted y values
    y_predict, test_losses, test_mse = test(model=model, loader=data_loader_test, loss_fn=loss_fn)

    actual = np.concatenate(np.array(dataset_test.y_data))
    predicted = np.concatenate(y_predict)
    # weights_rereversed = np.array(model.layer_1.data)[0, ::-1]
    weights_rereversed = model.layer_1.weight.detach().numpy()[0, ::-1]

    return predicted, actual, np.array(losses), weights_rereversed, test_mse, avg_losses


def run(data, model_config, train_config, verbose=False):
    if verbose:
        print("################ Model: AR-Net ################")
    start = time.time()
    predicted, actual, losses, weights, test_mse, epoch_losses = run_train_test(
        dataset_train=data["train"],
        dataset_test=data["test"],
        model_config=model_config,
        train_config=train_config,
        verbose=verbose,
    )
    end = time.time()
    duration = end - start

    if verbose:
        print("Time: {:8.4f}".format(duration))
        print("Final train epoch loss: {:10.2f}".format(epoch_losses[-1]))
        print("Test MSEs: {:10.2f}".format(test_mse))

    results = {}
    results["weights"] = weights
    results["predicted"] = predicted
    results["actual"] = actual
    results["test_mse"] = test_mse
    results["losses"] = losses
    results["epoch_losses"] = epoch_losses
    if data["type"] == 'AR':
        stats = utils.compute_stats_ar(results, ar_params=data["ar"], verbose=verbose)
    else:
        raise NotImplementedError
    stats["Time (s)"] = duration

    return results, stats


def main():
    print("deprecated")


if __name__ == "__main__":
    main()
