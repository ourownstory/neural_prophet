#!/usr/bin/env python3

import unittest
import os
import pathlib
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from neuralprophet import (
    NeuralProphet,
    df_utils,
    time_dataset,
    configure,
)

log = logging.getLogger("nprophet.test")
log.setLevel("WARNING")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "example_data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")


class UnitTests(unittest.TestCase):
    plot = False

    def test_impute_missing(
        self,
    ):
        """Debugging data preprocessing"""
        log.info("testing: Impute Missing")
        allow_missing_dates = False

        df = pd.read_csv(PEYTON_FILE)
        name = "test"
        df[name] = df["y"].values

        if not allow_missing_dates:
            df_na, _ = df_utils.add_missing_dates_nan(df.copy(deep=True), freq="D")
        else:
            df_na = df.copy(deep=True)
        to_fill = pd.isna(df_na["y"])
        # TODO fix debugging printout error
        log.debug("sum(to_fill): {}".format(sum(to_fill.values)))

        # df_filled, remaining_na = df_utils.fill_small_linear_large_trend(
        #     df.copy(deep=True),
        #     column=name,
        #     allow_missing_dates=allow_missing_dates
        # )
        df_filled, remaining_na = df_utils.fill_linear_then_rolling_avg(
            df.copy(deep=True), column=name, allow_missing_dates=allow_missing_dates
        )
        # TODO fix debugging printout error
        log.debug("sum(pd.isna(df_filled[name])): {}".format(sum(pd.isna(df_filled[name]).values)))

        if self.plot:
            if not allow_missing_dates:
                df, _ = df_utils.add_missing_dates_nan(df)
            df = df.loc[200:250]
            fig1 = plt.plot(df["ds"], df[name], "b-")
            fig1 = plt.plot(df["ds"], df[name], "b.")

            df_filled = df_filled.loc[200:250]
            # fig3 = plt.plot(df_filled['ds'], df_filled[name], 'kx')
            fig4 = plt.plot(df_filled["ds"][to_fill], df_filled[name][to_fill], "kx")
            plt.show()

    def test_time_dataset(self):
        # manually load any file that stores a time series, for example:
        df_in = pd.read_csv(AIR_FILE, index_col=False)
        log.debug("Infile shape: {}".format(df_in.shape))

        n_lags = 3
        n_forecasts = 1
        valid_p = 0.2
        df_train, df_val = df_utils.split_df(df_in, n_lags, n_forecasts, valid_p, inputs_overbleed=True)

        # create a tabularized dataset from time series
        df = df_utils.check_dataframe(df_train)
        data_params = df_utils.init_data_params(df, normalize="minmax")
        df = df_utils.normalize(df, data_params)
        inputs, targets = time_dataset.tabularize_univariate_datetime(
            df,
            n_lags=n_lags,
            n_forecasts=n_forecasts,
        )
        log.debug(
            "tabularized inputs: {}".format(
                "; ".join(["{}: {}".format(inp, values.shape) for inp, values in inputs.items()])
            )
        )

    def test_normalize(self):
        for add in [0, -1, 0.00000001, -0.99999999]:
            length = 1000
            days = pd.date_range(start="2017-01-01", periods=length)
            y = np.zeros(length)
            y[1] = 1
            y = y + add
            df = pd.DataFrame({"ds": days, "y": y})
            m = NeuralProphet(
                normalize="soft",
            )
            data_params = df_utils.init_data_params(
                df,
                normalize=m.normalize,
                covariates_config=m.config_covar,
                regressor_config=m.regressors_config,
                events_config=m.events_config,
            )
            df_norm = df_utils.normalize(df, data_params)

    def test_auto_batch_epoch(self):
        check = {
            "1": (1, 1000),
            "10": (2, 1000),
            "100": (8, 320),
            "1000": (32, 64),
            "10000": (128, 12),
            "100000": (128, 5),
        }
        for n_data in [1, 10, int(1e2), int(1e3), int(1e4), int(1e5)]:
            c = configure.Train(
                learning_rate=None, epochs=None, batch_size=None, loss_func="mse", ar_sparsity=None, train_speed=0
            )
            c.set_auto_batch_epoch(n_data)
            log.debug("n_data: {}, batch: {}, epoch: {}".format(n_data, c.batch_size, c.epochs))
            batch, epoch = check["{}".format(n_data)]
            assert c.batch_size == batch
            assert c.epochs == epoch

    def test_train_speed(self):
        df = pd.read_csv(PEYTON_FILE, nrows=102)[:100]
        batch_size = 16
        epochs = 2
        learning_rate = 1.0
        check = {
            "-2": (int(batch_size / 4), int(epochs * 4), learning_rate / 4),
            "-1": (int(batch_size / 2), int(epochs * 2), learning_rate / 2),
            "0": (batch_size, epochs, learning_rate),
            "1": (int(batch_size * 2), int(epochs / 2), learning_rate * 2),
            "2": (int(batch_size * 4), int(epochs / 4), learning_rate * 4),
        }
        for train_speed in [-1, 0, 1]:
            m = NeuralProphet(
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                train_speed=train_speed,
            )
            m.fit(df, freq="D")
            c = m.config_train
            log.debug(
                "train_speed: {}, batch: {}, epoch: {}, learning_rate: {}".format(
                    train_speed, c.batch_size, c.epochs, c.learning_rate
                )
            )
            batch, epoch, lr = check["{}".format(train_speed)]
            assert c.batch_size == batch
            assert c.epochs == epoch
            assert math.isclose(c.learning_rate, lr)

        batch_size = 8
        epochs = 320

        check2 = {
            "-2": (int(batch_size / 4), int(epochs * 4)),
            "-1": (int(batch_size / 2), int(epochs * 2)),
            "0": (batch_size, epochs),
            "1": (int(batch_size * 2), int(epochs / 2)),
            "2": (int(batch_size * 4), int(epochs / 4)),
        }
        for train_speed in [0, 1, 2]:
            m = NeuralProphet(
                train_speed=train_speed,
            )
            m.fit(df, freq="D")
            c = m.config_train
            log.debug("train_speed: {}, batch: {}, epoch: {}".format(train_speed, c.batch_size, c.epochs))
            batch, epoch = check2["{}".format(train_speed)]
            assert c.batch_size == batch
            assert c.epochs == epoch
