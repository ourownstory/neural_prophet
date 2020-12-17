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
import numpy as np
import torch
from torch import nn

log = logging.getLogger("NP.test")
log.setLevel("WARNING")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "example_data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
YOS_FILE = os.path.join(DATA_DIR, "yosemite_temps.csv")


class UnitTests(unittest.TestCase):
    plot = False

    def test_impute_missing(self):
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
        df_filled = df.copy(deep=True)
        df_filled.loc[:, name], remaining_na = df_utils.fill_linear_then_rolling_avg(
            df_filled[name], limit_linear=5, rolling=20
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

    def test_logistic_trend(self):
        log.info("testing: Logistic growth trend")

        t_min = 0
        t_max = 1
        samples = 20
        n_changepoints = 5

        torch.manual_seed(3)

        ds_freq = "H"

        idx = pd.date_range("2018-01-01", periods=samples, freq=ds_freq)

        t_datetime = pd.Series(idx)
        t = torch.linspace(0, 1, samples)

        df = pd.DataFrame()
        df["ds"] = t_datetime
        # dummy y
        df["y"] = t

        # target curves for testing:
        # 1. standard logistic curve with some abrupt changes in curvature, clearly discontinuous derivatives, cap/floor of model given (as in Prophet)
        # 2. logistic curve up and down with no flat section between, different initial rate, offset, floor, and cap, cap/floor of model given (as in Prophet)
        # 3. smooth logistic curve, different initial rate, offset, floor, and cap, cap/floor of model given (as in Prophet)
        # 4. same logistic curve as 3. with learned cap and floor (and with small regularization)
        trend_caps = [[-1.8455], [50.0], [5.0]]
        trend_floors = [[-20.3895], [5.0], [-25.0]]
        trend_k0s = [[40.5123], [24.5123], [100.0]]
        trend_deltas = [
            [12.2064, -35.0, 11.8366, 49.1343, -9.3666],
            [12.2064, 0.0, -150.0, 49.1343, -9.3666],
            [12.2064, -25.0, -160.0, 49.1343, -9.3666],
        ]
        trend_m0s = [[0.5992], [0.5], [0.2]]
        # whether to use target as cap/floor for testing user-set cap/floor
        correct_outputs = [
            [
                0.050763197243213654,
                0.07822930067777634,
                0.1059739962220192,
                0.1339426040649414,
                0.1865299940109253,
                0.2442164123058319,
                0.3016437590122223,
                0.3595620393753052,
                0.41680189967155457,
                0.47226324677467346,
                0.527276873588562,
                0.581228494644165,
                0.631985604763031,
                0.6845468878746033,
                0.7420904636383057,
                0.7932311296463013,
                0.8419517874717712,
                0.899278998374939,
                0.9445374011993408,
                0.9797079563140869,
            ],
            [
                0.06602450460195541,
                0.0857219472527504,
                0.10546455532312393,
                0.125232994556427,
                0.17851677536964417,
                0.23809069395065308,
                0.29672062397003174,
                0.35733675956726074,
                0.4174517095088959,
                0.47481510043144226,
                0.5319765210151672,
                0.5880259275436401,
                0.6395998597145081,
                0.6928597688674927,
                0.751064121723175,
                0.800636887550354,
                0.846593976020813,
                0.9001692533493042,
                0.9385048151016235,
                0.9654718637466431,
            ],
            [
                0.03325868397951126,
                0.06951137632131577,
                0.10665065795183182,
                0.144539475440979,
                0.19468414783477783,
                0.2478463053703308,
                0.3013042211532593,
                0.35797223448753357,
                0.41560980677604675,
                0.4719907343387604,
                0.5283746719360352,
                0.5840923190116882,
                0.6369045376777649,
                0.6904442310333252,
                0.7471389174461365,
                0.7982776761054993,
                0.8459429144859314,
                0.8965682983398438,
                0.9390300512313843,
                0.974210262298584,
            ],
        ]

        runs = len(trend_caps)

        for run in range(runs):
            model = NeuralProphet(
                growth="logistic",
                n_changepoints=n_changepoints,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                epochs=40,
            )
            model.fit(df, ds_freq, epochs=1)

            model.model.trend_cap = nn.Parameter(torch.Tensor(trend_caps[run]))
            model.model.trend_floor = nn.Parameter(torch.Tensor(trend_floors[run]))
            model.model.trend_k0 = nn.Parameter(torch.Tensor(trend_k0s[run]))
            model.model.trend_deltas = nn.Parameter(torch.Tensor(trend_deltas[run]))
            model.model.trend_m0 = nn.Parameter(torch.Tensor(trend_m0s[run]))
            model.fit(df, ds_freq)

            future = model.make_future_dataframe(df, periods=0, n_historic_predictions=len(df))
            pred = model.predict(future)["trend"]

            # print(pred)
            # assert np.allclose(list(pred), correct_outputs[run])

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
            "1": (int(batch_size * 2), max(1, int(epochs / 2)), learning_rate * 2),
            "2": (int(batch_size * 4), max(1, int(epochs / 4)), learning_rate * 4),
        }
        for train_speed in [-1, 0, 2]:
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
        for train_speed in [2]:
            m = NeuralProphet(
                train_speed=train_speed,
            )
            m.fit(df, freq="D")
            c = m.config_train
            log.debug("train_speed: {}, batch: {}, epoch: {}".format(train_speed, c.batch_size, c.epochs))
            batch, epoch = check2["{}".format(train_speed)]
            assert c.batch_size == batch
            assert c.epochs == epoch

    def test_split_impute(self):
        def check_split(df_in, df_len_expected, n_lags, n_forecasts, freq, p=0.1):
            m = NeuralProphet(
                n_lags=n_lags,
                n_forecasts=n_forecasts,
            )
            df_in = df_utils.check_dataframe(df_in, check_y=False)
            df_in = m._handle_missing_data(df_in, freq=freq, predicting=False)
            assert df_len_expected == len(df_in)

            total_samples = len(df_in) - n_lags - 2 * n_forecasts + 2
            df_train, df_test = m.split_df(df_in, freq=freq, valid_p=0.1, inputs_overbleed=True)
            n_train = len(df_train) - n_lags - n_forecasts + 1
            n_test = len(df_test) - n_lags - n_forecasts + 1
            assert total_samples == n_train + n_test

            n_test_expected = max(1, int(total_samples * p))
            n_train_expected = total_samples - n_test_expected
            assert n_train == n_train_expected
            assert n_test == n_test_expected

        log.info("testing: SPLIT: daily data")
        df = pd.read_csv(PEYTON_FILE)
        check_split(df_in=df, df_len_expected=len(df) + 59, freq="D", n_lags=10, n_forecasts=3)

        log.info("testing: SPLIT: monthly data")
        df = pd.read_csv(AIR_FILE)
        check_split(df_in=df, df_len_expected=len(df), freq="MS", n_lags=10, n_forecasts=3)

        log.info("testing: SPLIT:  5min data")
        df = pd.read_csv(YOS_FILE)
        check_split(df_in=df, df_len_expected=len(df), freq="5min", n_lags=10, n_forecasts=3)

        # redo with no lags
        log.info("testing: SPLIT: daily data")
        df = pd.read_csv(PEYTON_FILE)
        check_split(df_in=df, df_len_expected=len(df), freq="D", n_lags=0, n_forecasts=1)

        log.info("testing: SPLIT: monthly data")
        df = pd.read_csv(AIR_FILE)
        check_split(df_in=df, df_len_expected=len(df), freq="MS", n_lags=0, n_forecasts=1)

        log.info("testing: SPLIT:  5min data")
        df = pd.read_csv(YOS_FILE)
        check_split(df_in=df, df_len_expected=len(df) - 12, freq="5min", n_lags=0, n_forecasts=1)

    def test_cv(self):
        def check_folds(df, n_lags, n_forecasts, valid_fold_num, valid_fold_pct, fold_overlap_pct):

            folds = df_utils.crossvalidation_split_df(
                df, n_lags, n_forecasts, valid_fold_num, valid_fold_pct, fold_overlap_pct
            )

            train_folds_len = []
            val_folds_len = []
            for (f_train, f_val) in folds:
                train_folds_len.append(len(f_train))
                val_folds_len.append(len(f_val))

            train_folds_samples = [x - n_lags - n_forecasts + 1 for x in train_folds_len]
            val_folds_samples = [x - n_lags - n_forecasts + 1 for x in val_folds_len]
            total_samples = len(df) - n_lags - (2 * n_forecasts) + 2

            val_fold_each = max(1, int(total_samples * valid_fold_pct))
            overlap_each = int(fold_overlap_pct * val_fold_each)
            assert all([x == val_fold_each for x in val_folds_samples])
            train_folds_should = [
                total_samples - val_fold_each - (valid_fold_num - i - 1) * (val_fold_each - overlap_each)
                for i in range(valid_fold_num)
            ]
            assert all([x == y for (x, y) in zip(train_folds_samples, train_folds_should)])
            log.debug("total_samples: {}".format(total_samples))
            log.debug("val_fold_each: {}".format(val_fold_each))
            log.debug("overlap_each: {}".format(overlap_each))
            log.debug("val_folds_len: {}".format(val_folds_len))
            log.debug("val_folds_samples: {}".format(val_folds_samples))
            log.debug("train_folds_len: {}".format(train_folds_len))
            log.debug("train_folds_samples: {}".format(train_folds_samples))
            log.debug("train_folds_should: {}".format(train_folds_should))

        len_df = 100
        check_folds(
            df=pd.DataFrame({"ds": pd.date_range(start="2017-01-01", periods=len_df), "y": np.arange(len_df)}),
            n_lags=0,
            n_forecasts=1,
            valid_fold_num=3,
            valid_fold_pct=0.1,
            fold_overlap_pct=0.0,
        )
        len_df = 1000
        check_folds(
            df=pd.DataFrame({"ds": pd.date_range(start="2017-01-01", periods=len_df), "y": np.arange(len_df)}),
            n_lags=50,
            n_forecasts=10,
            valid_fold_num=10,
            valid_fold_pct=0.1,
            fold_overlap_pct=0.5,
        )
