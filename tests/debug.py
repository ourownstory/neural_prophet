#!/usr/bin/env python3

import os
import pathlib
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

from neuralprophet import (
    NeuralProphet,
    df_utils,
    time_dataset,
    configure,
)


log = logging.getLogger("NP.debug")
log.setLevel("INFO")
log.parent.setLevel("INFO")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
YOS_FILE = os.path.join(DATA_DIR, "yosemite_temps.csv")
NROWS = 256
EPOCHS = 2
BATCH_SIZE = 64

PLOT = False


def test_auto_batch_epoch():
    check = {
        "1": (1, 200),
        "10": (10, 200),
        "100": (16, 160),
        "1000": (32, 64),
        "10000": (64, 25),
        "100000": (128, 20),
        "1000000": (256, 20),
        "10000000": (512, 20),
    }

    observe = {}
    # for n_data in [10, int(1e3), int(1e6)]:
    for n_data, (batch_size, epochs) in check.items():
        n_data = int(n_data)
        c = configure.Train(
            learning_rate=None,
            epochs=None,
            batch_size=None,
            loss_func="mse",
            ar_sparsity=None,
            optimizer="SGD",
        )
        c.set_auto_batch_epoch(n_data=n_data)
        observe["{}".format(n_data)] = (c.batch_size, c.epochs)
        log.info("[config] n_data: {}, batch: {}, epoch: {}".format(n_data, c.batch_size, c.epochs))
        log.info("[should] n_data: {}, batch: {}, epoch: {}".format(n_data, batch_size, epochs))
        # assert c.batch_size == batch_size
        # assert c.epochs == epochs
    print(check)
    print(observe)


if __name__ == "__main__":
    test_auto_batch_epoch()
