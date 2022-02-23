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


if __name__ == "__main__":
    test_make_future()
