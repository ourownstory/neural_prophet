#!/usr/bin/env python3

import pytest
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
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
YOS_FILE = os.path.join(DATA_DIR, "yosemite_temps.csv")


plot = False


def test_impute_missing():
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
    if plot:
        if not allow_missing_dates:
            df, _ = df_utils.add_missing_dates_nan(df, freq="D")
        df = df.loc[200:250]
        fig1 = plt.plot(df["ds"], df[name], "b-")
        fig1 = plt.plot(df["ds"], df[name], "b.")
        df_filled = df_filled.loc[200:250]
        # fig3 = plt.plot(df_filled['ds'], df_filled[name], 'kx')
        fig4 = plt.plot(df_filled["ds"][to_fill], df_filled[name][to_fill], "kx")
        plt.show()


def test_time_dataset():
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


def test_normalize():
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


def test_add_lagged_regressors():
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
        metrics_df = m.fit(df1, freq="D", validation_df=df1[-100:])
        future = m.make_future_dataframe(df1, n_historic_predictions=365)
        ## Check if the future dataframe contains all the lagged regressors
        check = any(item in future.columns for item in cols)
        forecast = m.predict(future)
        log.debug(check)


def test_auto_batch_epoch():
    check = {
        "1": (1, 500),
        "10": (10, 500),
        "100": (16, 320),
        "1000": (32, 181),
        "10000": (64, 102),
        "100000": (128, 57),
        "1000000": (256, 50),
        "10000000": (256, 50),
    }
    for n_data in [10, int(1e3), int(1e6)]:
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


def test_train_speed_custom():
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


def test_train_speed_auto():
    df = pd.read_csv(PEYTON_FILE, nrows=102)[:100]
    batch_size = 16
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
        batch, epoch = check2["{}".format(train_speed)]
        log.debug("train_speed: {}, batch(check): {}, epoch(check): {}".format(train_speed, batch, epoch))
        log.debug("train_speed: {}, batch: {}, epoch: {}".format(train_speed, c.batch_size, c.epochs))
        assert c.batch_size == batch
        assert c.epochs == epoch


def test_split_impute():
    def check_split(df_in, df_len_expected, n_lags, n_forecasts, freq, p=0.1):
        m = NeuralProphet(
            n_lags=n_lags,
            n_forecasts=n_forecasts,
        )
        df_in = df_utils.check_dataframe(df_in, check_y=False)
        df_in = m.handle_missing_data(df_in, freq=freq, predicting=False)
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


def test_cv():
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


def test_reg_delay():
    df = pd.read_csv(PEYTON_FILE, nrows=102)[:100]
    m = NeuralProphet(epochs=10)
    m.fit(df, freq="D")
    c = m.config_train
    for w, e, i in [
        (0, 0, 1),
        (0, 3, 0),
        (0, 5, 0),
        # (0.002739052315863355, 5, 0.1),
        (0.5, 6, 0.5),
        # (0.9972609476841366, 7, 0.9),
        (1, 7, 1),
        (1, 8, 0),
    ]:
        weight = c.get_reg_delay_weight(e, i, reg_start_pct=0.5, reg_full_pct=0.8)
        log.debug("e {}, i {}, expected w {}, got w {}".format(e, i, w, weight))
        assert weight == w


def test_double_crossvalidation():
    len_df = 100
    folds_val, folds_test = df_utils.double_crossvalidation_split_df(
        df=pd.DataFrame({"ds": pd.date_range(start="2017-01-01", periods=len_df), "y": np.arange(len_df)}),
        n_lags=0,
        n_forecasts=1,
        k=3,
        valid_pct=0.3,
        test_pct=0.15,
    )
    train_folds_len1 = []
    val_folds_len1 = []
    for (f_train, f_val) in folds_val:
        train_folds_len1.append(len(f_train))
        val_folds_len1.append(len(f_val))
    train_folds_len2 = []
    val_folds_len2 = []
    for (f_train, f_val) in folds_test:
        train_folds_len2.append(len(f_train))
        val_folds_len2.append(len(f_val))
    assert train_folds_len1[-1] == 75
    assert train_folds_len2[0] == 85
    assert val_folds_len1[0] == 10
    assert val_folds_len2[0] == 5
    log.debug("train_folds_len1: {}".format(train_folds_len1))
    log.debug("val_folds_len1: {}".format(val_folds_len1))
    log.debug("train_folds_len2: {}".format(train_folds_len2))
    log.debug("val_folds_len2: {}".format(val_folds_len2))


def test_check_duplicate_ds():
    # Check whether a ValueError is thrown in case there
    # are duplicate dates in the ds column of dataframe
    df = pd.read_csv(PEYTON_FILE, nrows=102)[:50]
    # introduce duplicates in dataframe
    df = pd.concat([df, df[8:9]]).reset_index()
    # Check if error thrown on duplicates
    m = NeuralProphet(
        n_lags=24,
        ar_sparsity=0.5,
    )
    with pytest.raises(ValueError):
        m.fit(df, freq="D")


def test_infer_frequency():
    df = pd.read_csv(PEYTON_FILE, nrows=102)[:50]
    m = NeuralProphet()
    # Check if freq is set automatically
    df_train, df_test = m.split_df(df)
    log.debug("freq automatically set")
    # Check if freq is set when equal to the original
    df_train, df_test = m.split_df(df, freq="D")
    log.debug("freq is equal to ideal")
    # Check if freq is set in different freq
    df_train, df_test = m.split_df(df, freq="5D")
    log.debug("freq is set even though is different than the ideal")
    # Assert for data unevenly spaced
    index = np.unique(np.geomspace(1, 40, 20, dtype=int))
    df_uneven = df.iloc[index, :]
    with pytest.raises(ValueError):
        m.split_df(df_uneven)
    # Check if freq is set even in a df with multiple freqs
    df_train, df_test = m.split_df(df_uneven, freq="H")
    log.debug("freq is set even with not definable freq")
    # Check if freq is set for list
    df_list = list((df, df))
    m = NeuralProphet()
    m.fit(df_list, epochs=5)
    log.debug("freq is set for list of dataframes")
    # Check if freq is set for list with different freq for n_lags=0
    df1 = df.copy(deep=True)
    time_range = pd.date_range(start="1994-12-01", periods=df.shape[0], freq="M")
    df1["ds"] = time_range
    df_list = list((df, df1))
    m = NeuralProphet(n_lags=0, epochs=5)
    m.fit(df_list, epochs=5)
    log.debug("freq is set for list of dataframes(n_lags=0)")
    # Assert for automatic frequency in list with different freq
    m = NeuralProphet(n_lags=2)
    with pytest.raises(ValueError):
        m.fit(df_list, epochs=5)
    # Exceptions
    frequencies = ["M", "MS", "Y", "YS", "Q", "QS", "B", "BH"]
    df = df.iloc[:200, :]
    for freq in frequencies:
        df1 = df.copy(deep=True)
        time_range = pd.date_range(start="1994-12-01", periods=df.shape[0], freq=freq)
        df1["ds"] = time_range
        df_train, df_test = m.split_df(df1)
    log.debug("freq is set for all the exceptions")
