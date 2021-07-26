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
                df, _ = df_utils.add_missing_dates_nan(df, freq="D")
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
        df_train, df_val = df_utils.split_df(df_in, n_lags, n_forecasts, valid_p)

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

    def test_add_lagged_regressors(self):
        NROWS = 512
        EPOCHS = 3
        BATCH_SIZE = 32

        df = pd.read_csv(PEYTON_FILE, nrows=NROWS)

        df["A"] = df["y"].rolling(7, min_periods=1).mean()
        df["B"] = df["y"].rolling(15, min_periods=1).mean()
        df["C"] = df["y"].rolling(30, min_periods=1).mean()

        col_dict = {
            "1": "A",
            "2": ["B"],
            "3": ["A", "B", "C"],
        }

        for key, value in col_dict.items():
            log.debug(value)
            if isinstance(value, list):
                feats = np.array(["ds", "y"] + value)
            else:
                feats = np.array(["ds", "y", value])
            df1 = pd.DataFrame(df, columns=feats)
            cols = [col for col in df1.columns if col not in ["ds", "y"]]
            m = NeuralProphet(
                n_forecasts=1,
                n_lags=3,
                weekly_seasonality=False,
                daily_seasonality=False,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
            )
            m = m.add_lagged_regressor(names=cols)
            metrics_df = m.fit(df1, freq="D", validate_each_epoch=True)
            future = m.make_future_dataframe(df1, n_historic_predictions=365)
            ## Check if the future dataframe contains all the lagged regressors
            check = any(item in future.columns for item in cols)
            forecast = m.predict(future)
            log.debug(check)

    def test_auto_batch_epoch(self):
        n2b = lambda x: int(400 / (1 + np.log(x / 100)))
        check = {
            "3": (3, 400),
            "10": (10, 400),
            "30": (16, 400),
            "100": (16, 400),
            "300": (16, 190),
            "1000": (32, 121),
            "10000": (64, 71),
            "100000": (128, 50),
            "1000000": (256, 40),
            "10000000": (256, 40),
        }
        for n_data in [3, 10, 30, int(1e2), int(1e3), int(1e4), int(1e5), int(1e6), int(1e7)]:
            c = configure.Train(
                learning_rate=None,
                epochs=None,
                batch_size=None,
                loss_func="mse",
                ar_sparsity=None,
                train_speed=0,
                optimizer="SGD",
            )
            c.set_auto_batch_epoch(n_data)
            log.debug("n_data: {}, batch: {}, epoch: {}".format(n_data, c.batch_size, c.epochs))
            batch, epoch = check["{}".format(n_data)]
            assert c.batch_size == batch
            assert c.epochs == epoch

    def test_train_speed_custom(self):
        df = pd.read_csv(PEYTON_FILE, nrows=102)[:100]
        batch_size = 16
        epochs = 4
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

    def test_train_speed_auto(self):
        df = pd.read_csv(PEYTON_FILE, nrows=102)[:100]
        batch_size = 16
        epochs = 400
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
            batch, epoch = check2["{}".format(train_speed)]
            log.debug("train_speed: {}, batch(check): {}, epoch(check): {}".format(train_speed, batch, epoch))
            log.debug("train_speed: {}, batch: {}, epoch: {}".format(train_speed, c.batch_size, c.epochs))

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
            df_train, df_test = m.split_df(df_in, freq=freq, valid_p=0.1)
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

    def test_reg_delay(self):
        df = pd.read_csv(PEYTON_FILE, nrows=102)[:100]
        m = NeuralProphet(epochs=10)
        m.fit(df, freq="D")
        c = m.config_train
        for w, e, i in [
            (0, 0, 1),
            (0, 3, 0),
            (0, 5, 0),
            (0.002739052315863355, 5, 0.1),
            (0.5, 6, 0.5),
            (0.9972609476841366, 7, 0.9),
            (1, 7, 1),
            (1, 8, 0),
        ]:
            weight = c.get_reg_delay_weight(e, i, reg_start_pct=0.5, reg_full_pct=0.8)
            log.debug("e {}, i {}, expected w {}, got w {}".format(e, i, w, weight))
            assert weight == w

    def test_global_modeling(self):
        log.debug('Testing GLOBAL MODELING')
        df = pd.read_csv(PEYTON_FILE)
        df1=df.iloc[:600,:].copy(deep=True)
        df2=df.iloc[600:1200,:].copy(deep=True)
        df3=df.iloc[1200:1800,:].copy(deep=True)
        df4=df.iloc[1800:2400,:].copy(deep=True)
        df1_0=df1.copy(deep=True)
        df2_0=df2.copy(deep=True)
        df3_0=df3.copy(deep=True)
        df4_0=df4.copy(deep=True)
        def gm():
            ## Testing df train / df test - no events, no regressors
            log.debug('Testing df train / df test - no events, no regressors')
            m = NeuralProphet(n_forecasts=2,n_lags=10,yearly_seasonality=False,
            weekly_seasonality=False,daily_seasonality=False)
            metrics = m.fit(df1_0, freq="D")
            future = m.make_future_dataframe(df2_0,n_historic_predictions=True)
            forecast = m.predict(df=future)
            fig=m.plot(forecast) 
            fig=m.plot_components(forecast)
            ## Testing LIST df train / df test - no events, no regressors
            log.debug('Testing LIST df train / df test - no events, no regressors')
            DF_Train=list()
            DF_Train.append(df1_0)
            DF_Train.append(df2_0)
            m = NeuralProphet(n_forecasts=2,n_lags=10,yearly_seasonality=False,
            weekly_seasonality=False,daily_seasonality=False)
            metrics = m.fit(DF_Train, freq="D")
            future = m.make_future_dataframe(df3_0,n_historic_predictions=True)
            forecast = m.predict(df=future)
            fig=m.plot(forecast) 
            fig=m.plot_components(forecast)
            ## ## Testing LIST df train / LIST df test - no events, no regressors
            log.debug('Testing LIST df train / LIST df test - no events, no regressors')
            DF_Test=list()
            DF_Test.append(df3_0)
            DF_Test.append(df4_0)
            m = NeuralProphet(n_forecasts=2,n_lags=10,yearly_seasonality=False,
            weekly_seasonality=False,daily_seasonality=False)
            metrics = m.fit(DF_Train, freq="D")
            future = m.make_future_dataframe(DF_Test,n_historic_predictions=True)
            forecast = m.predict(df=future)
            fig=m.plot(forecast) 
            for frst in forecast:
                fig=m.plot_components(frst)  #Notice that the plot forecast does not accept lists
        #---- REGRESSORS ----
        def gm_regressors():
            log.debug('REGRESSORS')
            df1['A'] = df1['y'].rolling(30, min_periods=1).mean()
            df2['A'] = df2['y'].rolling(10, min_periods=1).mean()
            df3['A'] = df3['y'].rolling(40, min_periods=1).mean()
            df4['A'] = df4['y'].rolling(20, min_periods=1).mean()
            future_regressors_df1 = pd.DataFrame(data={'A': df1['A'][:25]})
            future_regressors_df2 = pd.DataFrame(data={'A': df2['A'][:30]})
            future_regressors_df3 = pd.DataFrame(data={'A': df3['A'][:30]})
            future_regressors_df4 = pd.DataFrame(data={'A': df4['A'][:40]})
            ## Testing df train / df test - df regressor, no events
            log.debug('Testing df train / df test - df regressor, no events')
            m = NeuralProphet(n_forecasts=2,n_lags=10,yearly_seasonality=False,
            weekly_seasonality=False,daily_seasonality=False)
            m = m.add_future_regressor(name='A')
            metrics = m.fit(df1, freq="D")
            future = m.make_future_dataframe(df2,n_historic_predictions=True,regressors_df=future_regressors_df2)
            forecast = m.predict(df=future)
            fig=m.plot(forecast) 
            fig=m.plot_components(forecast)
            ## Testing LIST df train / df test - df regressors, no events
            log.debug('Testing LIST df train / df test - df regressors, no events')
            DF_Train=list()
            DF_Train.append(df1)
            DF_Train.append(df2)
            m = NeuralProphet(n_forecasts=2,n_lags=10,yearly_seasonality=False,
            weekly_seasonality=False,daily_seasonality=False)
            m = m.add_future_regressor(name='A')
            metrics = m.fit(DF_Train, freq="D")
            future = m.make_future_dataframe(df3,n_historic_predictions=True,regressors_df=future_regressors_df3)
            forecast = m.predict(df=future)
            fig=m.plot(forecast) 
            fig=m.plot_components(forecast)
            ## Testing LIST df train / LIST df test - df regressors, no events
            log.debug('Testing LIST df train / LIST df test - df regressors, no events')
            DF_Test=list()
            DF_Test.append(df3)
            DF_Test.append(df4)
            m = NeuralProphet(n_forecasts=2,n_lags=10,yearly_seasonality=False,
            weekly_seasonality=False,daily_seasonality=False)
            m = m.add_future_regressor(name='A')
            metrics = m.fit(DF_Train, freq="D")
            future = m.make_future_dataframe(DF_Test,n_historic_predictions=True,regressors_df=future_regressors_df3)
            forecast = m.predict(df=future)
            fig=m.plot(forecast) 
            for frst in forecast:
                fig=m.plot_components(frst)
            ##Testing LIST df train / LIST df test - LIST regressors, no events
            log.debug('Testing LIST df train / LIST df test - LIST regressors, no events')
            DF_FR=list()
            DF_FR.append(future_regressors_df3)
            DF_FR.append(future_regressors_df4)
            m = NeuralProphet(n_forecasts=2,n_lags=10,yearly_seasonality=False,
            weekly_seasonality=False,daily_seasonality=False)
            m = m.add_future_regressor(name='A')
            metrics = m.fit(DF_Train, freq="D")
            future = m.make_future_dataframe(DF_Test,n_historic_predictions=True,regressors_df=DF_FR)
            forecast = m.predict(df=future)
            fig=m.plot(forecast) 
            for frst in forecast:
                fig=m.plot_components(frst)
        #---- EVENTS ----
        def gm_events():
            log.debug('EVENTS')
            playoffs_history = pd.DataFrame({
            'event': 'playoff',
            'ds': pd.to_datetime(['2007-12-13', '2009-08-23', '2009-08-25',
                                '2009-09-02', '2011-05-09', '2011-05-11',
                                '2011-05-16', '2013-01-01', '2013-01-05',
                                '2013-01-10', '2014-08-27', '2014-08-31'])})
            history_events_df1 = playoffs_history.iloc[:3,:].copy(deep=True)
            history_events_df2 = playoffs_history.iloc[3:6,:].copy(deep=True)
            history_events_df3 = playoffs_history.iloc[6:9,:].copy(deep=True)
            history_events_df4 = playoffs_history.iloc[9:,:].copy(deep=True)
            playoffs_future = pd.DataFrame({
            'event': 'playoff',
            'ds': pd.to_datetime(['2009-08-27', '2009-08-29', '2011-05-14','2011-05-17',
            '2013-01-10', '2013-01-15', '2014-09-05','2014-09-08'])})
            future_events_df1 = playoffs_future.iloc[:2,:].copy(deep=True)
            future_events_df2 = playoffs_future.iloc[2:4,:].copy(deep=True)
            future_events_df3 = playoffs_future.iloc[4:6,:].copy(deep=True)
            future_events_df4 = playoffs_future.iloc[6:8,:].copy(deep=True)
            ## Testing df train / df test - df events, no regressors
            log.debug('Testing df train / df test - df events, no regressors')
            m = NeuralProphet(n_lags=10,n_forecasts=5,yearly_seasonality=False,
            weekly_seasonality=False,daily_seasonality=False)
            m = m.add_events(["playoff"])
            history_df1 = m.create_df_with_events(df1_0,history_events_df1)
            history_df2 = m.create_df_with_events(df2_0,history_events_df2)
            metrics = m.fit(history_df1, freq="D")
            future = m.make_future_dataframe(history_df2,n_historic_predictions=True,events_df=future_events_df1)
            forecast = m.predict(future)
            fig=m.plot(forecast) 
            fig=m.plot_components(forecast)
            ## Testing LIST train / df test - df events, no regressors
            log.debug('Testing LIST train / df test - df events, no regressors')
            m = NeuralProphet(n_lags=10,n_forecasts=5,yearly_seasonality=False,
            weekly_seasonality=False,daily_seasonality=False)
            m = m.add_events(["playoff"])
            history_df1 = m.create_df_with_events(df1_0,history_events_df1)
            history_df2 = m.create_df_with_events(df2_0,history_events_df2)
            history_df3 = m.create_df_with_events(df3_0,history_events_df3)
            DF_Train=list()
            DF_Train.append(history_df1)
            DF_Train.append(history_df2)
            metrics = m.fit(DF_Train, freq="D")
            future = m.make_future_dataframe(history_df3,n_historic_predictions=True,events_df=future_events_df3)
            forecast = m.predict(future)
            fig=m.plot(forecast) 
            fig=m.plot_components(forecast)
            ## Testing LIST train / LIST test - df events, no regressors
            log.debug('Testing LIST train / LIST test - df events, no regressors')
            m = NeuralProphet(n_lags=10,n_forecasts=5,yearly_seasonality=False,
            weekly_seasonality=False,daily_seasonality=False)
            m = m.add_events(["playoff"])
            history_df1 = m.create_df_with_events(df1_0,history_events_df1)
            history_df2 = m.create_df_with_events(df2_0,history_events_df2)
            history_df3 = m.create_df_with_events(df3_0,history_events_df3)
            history_df4 = m.create_df_with_events(df4_0,history_events_df4)
            DF_Train=list()
            DF_Train.append(history_df1)
            DF_Train.append(history_df2)
            DF_Test=list()
            DF_Test.append(history_df3)
            DF_Test.append(history_df4)
            metrics = m.fit(DF_Train, freq="D")
            future = m.make_future_dataframe(DF_Test,n_historic_predictions=True,events_df=future_events_df4)
            forecast = m.predict(future)
            fig=m.plot(forecast) 
            for frst in forecast:
                fig=m.plot_components(frst)
            ## Testing LIST train / LIST test - LIST events, no regressors
            log.debug('Testing LIST train / LIST test - LIST events, no regressors')
            DF_FE=list()
            DF_FE.append(future_events_df3)
            DF_FE.append(future_events_df4)
            m = NeuralProphet(n_lags=10,n_forecasts=5,yearly_seasonality=False,
            weekly_seasonality=False,daily_seasonality=False)
            m = m.add_events(["playoff"])
            history_df1 = m.create_df_with_events(df1_0,history_events_df1)
            history_df2 = m.create_df_with_events(df2_0,history_events_df2)
            history_df3 = m.create_df_with_events(df3_0,history_events_df3)
            history_df4 = m.create_df_with_events(df4_0,history_events_df4)
            DF_Train=list()
            DF_Train.append(history_df1)
            DF_Train.append(history_df2)
            DF_Test=list()
            DF_Test.append(history_df3)
            DF_Test.append(history_df4)
            metrics = m.fit(DF_Train, freq="D")
            future = m.make_future_dataframe(DF_Test,n_historic_predictions=True,events_df=DF_FE)
            forecast = m.predict(future)
            fig=m.plot(forecast) 
            for frst in forecast:
                fig=m.plot_components(frst)
        ## EVENTS + REGRESSORS
        def gm_events_plus_regressors():
            log.debug('EVENTS + REGRESSORS')
            m = NeuralProphet(n_lags=10,n_forecasts=5,yearly_seasonality=False,
            weekly_seasonality=False,daily_seasonality=False)
            m = m.add_events(["playoff"])
            m = m.add_future_regressor(name='A')
            history_df1 = m.create_df_with_events(df1,history_events_df1)
            history_df2 = m.create_df_with_events(df2,history_events_df2)
            history_df3 = m.create_df_with_events(df3,history_events_df3)
            history_df4 = m.create_df_with_events(df4,history_events_df4)
            DF_Train=list()
            DF_Train.append(history_df1)
            DF_Train.append(history_df2)
            DF_Test=list()
            DF_Test.append(history_df3)
            DF_Test.append(history_df4)
            metrics = m.fit(DF_Train, freq="D")
            future = m.make_future_dataframe(DF_Test,n_historic_predictions=True,
            events_df=DF_FE, regressors_df=DF_FR)
            forecast = m.predict(future)
            fig=m.plot(forecast) 
            for frst in forecast:
                fig=m.plot_components(frst)
        gm()
        gm_regressors()
        gm_events()
        gm_events_plus_regressors()
        log.debug('GLOBAL MODELING TESTING - DONE')