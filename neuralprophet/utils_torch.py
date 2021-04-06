import numpy as np
import logging
import torch
from torch.utils.data import DataLoader, Subset
import inspect
from torch_lr_finder import LRFinder

from neuralprophet import utils

log = logging.getLogger("NP.utils_torch")


def lr_range_test(
    model,
    dataset,
    loss_func,
    optimizer="AdamW",
    batch_size=32,
    num_iter=None,
    skip_start=10,
    skip_end=10,
    start_lr=1e-7,
    end_lr=10,
    plot=False,
):
    if num_iter is None:
        num_iter = 100 + int(np.log10(10 + len(dataset)) * 50)
    n_train = min(len(dataset), num_iter * batch_size)
    n_val = min(int(0.3 * len(dataset)), 2 * num_iter)
    log.debug("num_iter: {}, n_val: {}".format(num_iter, n_val))
    split_idx = int(0.7 * len(dataset))
    idx_train = np.random.choice(split_idx, size=n_train)
    idx_val = np.random.choice(np.arange(split_idx, len(dataset)), size=n_val)
    train_data = Subset(dataset, idx_train)
    val_data = Subset(dataset, idx_val)
    lrtest_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    lrtest_loader_val = DataLoader(val_data, batch_size=1024, shuffle=True)
    lrtest_optimizer = create_optimizer(optimizer, model.parameters(), start_lr)
    with utils.HiddenPrints():
        lr_finder = LRFinder(model, lrtest_optimizer, loss_func)
        lr_finder.range_test(
            lrtest_loader,
            val_loader=lrtest_loader_val,
            end_lr=end_lr,
            num_iter=num_iter,
            smooth_f=0.2,  # re-consider if lr-rate varies a lot
        )
        lrs = lr_finder.history["lr"]
        losses = lr_finder.history["loss"]
    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]
    if plot:
        with utils.HiddenPrints():
            ax, steepest_lr = lr_finder.plot()  # to inspect the loss-learning rate graph
    max_lr = None
    try:
        steep_idx = (np.gradient(np.array(losses))).argmin()
        min_idx = (np.array(losses)).argmin()
        steep_lr = lrs[steep_idx]
        min_lr = lrs[min_idx]
        max_lr = 10 ** ((np.log10(steep_lr) + 2.0 * np.log10(min_lr)) / 3.0)
        log.info("lr-range-test results: steep: {:.2E}, min: {:.2E}".format(steep_lr, min_lr))
    except ValueError:
        log.error("Failed to compute the gradients, there might not be enough points.")
    if max_lr is not None:
        log.info("learning rate range test selected lr: {:.2E}".format(max_lr))
    else:
        max_lr = 0.1
        log.error("lr range test failed. defaulting to lr: {}".format(max_lr))
    with utils.HiddenPrints():
        lr_finder.reset()  # to reset the model and optimizer to their initial state
    return max_lr


def create_optimizer(optimizer, model_parameters, lr):
    if type(optimizer) == str:
        if optimizer.lower() == "adamw":
            # Tends to overfit, but reliable
            optimizer = torch.optim.AdamW(model_parameters, lr=lr, weight_decay=1e-3)
        elif optimizer.lower() == "sgd":
            # better validation performance, but diverges sometimes
            optimizer = torch.optim.SGD(model_parameters, lr=lr, momentum=0.9)  # weight_decay=1e-5
        else:
            raise ValueError
    elif inspect.isclass(optimizer) and issubclass(optimizer, torch.optim.Optimizer):
        optimizer = optimizer(model_parameters, lr=lr)
    else:
        raise ValueError
    return optimizer
