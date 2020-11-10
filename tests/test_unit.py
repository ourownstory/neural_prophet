#!/usr/bin/env python3

import unittest
import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import logging
from neuralprophet import (
    NeuralProphet,
    df_utils,
    time_dataset,
)
import numpy as np
import torch
from torch import nn

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

    def test_time_dataset(
        self,
    ):
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

        idx = pd.date_range('2018-01-01', periods=samples, freq=ds_freq)

        t_datetime = pd.Series(idx)
        t = torch.linspace(0, 1, samples)

        df = pd.DataFrame()
        df['ds'] = t_datetime
        # dummy y
        df['y'] = t

        # target curves for testing:
        # 1. standard logistic curve with some abrupt changes in curvature, clearly discontinuous derivatives, cap/floor of model given (as in Prophet)
        # 2. logistic curve up and down with no flat section between, different initial rate, offset, floor, and cap, cap/floor of model given (as in Prophet)
        # 3. smooth logistic curve, different initial rate, offset, floor, and cap, cap/floor of model given (as in Prophet)
        # 4. same logistic curve as 3. with learned cap and floor (and with small regularization)
        trend_caps = [[-1.8455], [50.0], [5.0]]
        trend_floors = [[-20.3895], [5.0], [-25.0]]
        trend_k0s = [[40.5123], [24.5123], [100.0]]
        trend_deltas = [[12.2064, -35.0,  11.8366,  49.1343,  -9.3666],
                        [12.2064, 0.0,  -150.0,  49.1343,  -9.3666],
                        [12.2064, -25.0,  -160.0,  49.1343,  -9.3666]]
        trend_m0s = [[0.5992], [0.5], [0.2]]
        # whether to use target as cap/floor for testing user-set cap/floor
        correct_outputs = [[0.3702579140663147, 0.3795645833015442, 0.3889691233634949, 0.39846253395080566, 0.4078933596611023, 
                            0.417363703250885, 0.42689239978790283, 0.4366762638092041, 0.44660407304763794, 0.4565622806549072, 
                            0.464622437953949, 0.4707714319229126, 0.4769214987754822, 0.48332029581069946, 0.4902138113975525, 
                            0.4970986247062683, 0.5045955777168274, 0.5151849985122681, 0.5257216095924377, 0.5361924767494202],
                           [0.4096601605415344, 0.4193142056465149, 0.42902672290802, 0.438787579536438, 0.45016586780548096, 
                            0.46189528703689575, 0.4736432433128357, 0.4857770800590515, 0.4980834126472473, 0.5103479623794556, 
                            0.5231962203979492, 0.5365920662879944, 0.5498590469360352, 0.5628329515457153, 0.575359046459198, 
                            0.5876969695091248, 0.5997359156608582, 0.6111056804656982, 0.6222555041313171, 0.6331722140312195],
                           [0.4133463501930237, 0.42302393913269043, 0.4327561855316162, 0.4425327777862549, 0.45146113634109497, 
                            0.46023279428482056, 0.46901625394821167, 0.4787430763244629, 0.48893362283706665, 0.49910634756088257, 
                            0.5102629661560059, 0.5223758816719055, 0.5344076156616211, 0.5457048416137695, 0.5556398630142212, 
                            0.5654807686805725, 0.5752135515213013, 0.5848125219345093, 0.5942888855934143, 0.6036339402198792]]

        runs = len(trend_caps)

        for run in range(runs):
            model = NeuralProphet(
                growth='logistic',
                n_changepoints=n_changepoints,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
            )
            model.model = model._init_model()
            model.model.trend_cap = nn.Parameter(torch.Tensor(trend_caps[run]))
            model.model.trend_floor = nn.Parameter(torch.Tensor(trend_floors[run]))
            model.model.trend_k0 = nn.Parameter(torch.Tensor(trend_k0s[run]))
            model.model.trend_deltas = nn.Parameter(torch.Tensor(trend_deltas[run]))
            model.model.trend_m0 = nn.Parameter(torch.Tensor(trend_m0s[run]))
            model.train_config['epochs'] = 0
            model.fit(df, ds_freq)

            future = model.make_future_dataframe(df, future_periods=0, n_historic_predictions=len(df))
            pred = model.predict(future)['trend']

            assert np.allclose(list(pred), correct_outputs[run])
