import logging
import os
import pathlib
import time
from itertools import product

import pandas as pd
import torch.utils.benchmark as benchmark
from torch.utils.data import DataLoader

from neuralprophet import NeuralProphet, df_utils, utils
from neuralprophet.data.process import _check_dataframe, _create_dataset, _handle_missing_data
from neuralprophet.data.transform import _normalize

# from neuralprophet.forecaster import

log = logging.getLogger("NP.test")
# log.setLevel("INFO")
# log.parent.setLevel("INFO")
# log.setLevel("WARNING")
# log.parent.setLevel("WARNING")
log.setLevel("ERROR")
log.parent.setLevel("ERROR")

DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
YOS_FILE = os.path.join(DATA_DIR, "yosemite_temps.csv")
NROWS = 1000
EPOCHS = 1
BATCH_SIZE = 10
LR = 1.0


def print_input_shapes(inputs):
    tabularized_input_shapes_str = ""
    for key, value in inputs.items():
        if key in [
            "seasonalities",
            "covariates",
            "events",
            "regressors",
        ]:
            for name, period_features in value.items():
                tabularized_input_shapes_str += f"    {name} {key} {period_features.shape}\n"
        else:
            tabularized_input_shapes_str += f"    {key} {value.shape} \n"
    print(f"Tabularized inputs shapes: \n{tabularized_input_shapes_str}")


def load(nrows=NROWS, epochs=EPOCHS, batch=BATCH_SIZE, season=True, iterations=1):
    tic = time.perf_counter()
    df = pd.read_csv(YOS_FILE, nrows=nrows)
    freq = "5min"
    num_workers = 0

    m = NeuralProphet(
        n_lags=12,
        n_forecasts=6,
        epochs=epochs,
        batch_size=batch,
        learning_rate=LR,
        yearly_seasonality=season,
        weekly_seasonality=season,
        daily_seasonality=season,
    )

    # Mimick m.fit(df) behavior

    df, _, _, m.id_list = df_utils.prep_or_copy_df(df)
    df = _check_dataframe(m, df, check_y=True, exogenous=True)
    m.data_freq = df_utils.infer_frequency(df, n_lags=m.max_lags, freq=freq)
    df = _handle_missing_data(
        df=df,
        freq=m.data_freq,
        n_lags=m.n_lags,
        n_forecasts=m.n_forecasts,
        config_missing=m.config_missing,
        config_regressors=m.config_regressors,
        config_lagged_regressors=m.config_lagged_regressors,
        config_events=m.config_events,
        config_seasonality=m.config_seasonality,
        predicting=False,
    )
    # mimick _init_train_loader
    m.config_normalization.init_data_params(
        df=df,
        config_lagged_regressors=m.config_lagged_regressors,
        config_regressors=m.config_regressors,
        config_events=m.config_events,
        config_seasonality=m.config_seasonality,
    )
    df = _normalize(df=df, config_normalization=m.config_normalization)

    df_merged = df_utils.merge_dataframes(df)
    m.config_seasonality = utils.set_auto_seasonalities(df_merged, config_seasonality=m.config_seasonality)
    if m.config_country_holidays is not None:
        m.config_country_holidays.init_holidays(df_merged)

    dataset = _create_dataset(
        m, df, predict_mode=False, prediction_frequency=m.prediction_frequency
    )  # needs to be called after set_auto_seasonalities

    # Determine the max_number of epochs
    m.config_train.set_auto_batch_epoch(n_data=len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=m.config_train.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    # dataset_size = len(df)
    # print(dataset_size)

    dataloader_iterator = iter(loader)
    toc = time.perf_counter()
    print(f"######## Time: {toc - tic:0.4f} for setup")
    tic = time.perf_counter()
    for i in range(iterations):
        data, target, meta = next(dataloader_iterator)
        # try:
        #     data, target, meta = next(dataloader_iterator)
        # except StopIteration:
        #     dataloader_iterator = iter(loader)
        #     data, target, meta = next(dataloader_iterator)
        # do_something()
    toc = time.perf_counter()
    # print_input_shapes(data)
    # print(len(meta["df_name"]))
    print(f"######## Time: {toc - tic:0.4f} for iterating {iterations} batches of size {batch}")


load(nrows=1010, batch=100, iterations=10)


def yosemite(nrows=NROWS, epochs=EPOCHS, batch=BATCH_SIZE, season=True):
    # log.info("testing: Uncertainty Estimation Yosemite Temps")
    df = pd.read_csv(YOS_FILE, nrows=nrows)
    m = NeuralProphet(
        n_lags=12,
        n_forecasts=6,
        quantiles=[0.01, 0.99],
        epochs=epochs,
        batch_size=batch,
        learning_rate=LR,
        yearly_seasonality=season,
        weekly_seasonality=season,
        daily_seasonality=season,
    )
    # tic = time.perf_counter()
    m.fit(df, freq="5min")
    # toc = time.perf_counter()
    # print(f"######## Time: {toc - tic:0.4f} for fit")

    # tic = time.perf_counter()
    # future = m.make_future_dataframe(df, periods=6, n_historic_predictions=3 * 24 * 12)
    # toc = time.perf_counter()
    # print(f"######## Time: {toc - tic:0.4f} for make_future_dataframe")

    # tic = time.perf_counter()
    # m.predict(future)
    # toc = time.perf_counter()
    # print(f"######## Time: {toc - tic:0.4f} for predict")

    m.highlight_nth_step_ahead_of_each_forecast(m.n_forecasts)


def peyton(nrows=NROWS, epochs=EPOCHS, batch=BATCH_SIZE, season=True):
    # log.info("testing: Uncertainty Estimation Peyton Manning")
    df = pd.read_csv(PEYTON_FILE, nrows=nrows)
    playoffs = pd.DataFrame(
        {
            "event": "playoff",
            "ds": pd.to_datetime(
                [
                    "2008-01-13",
                    "2009-01-03",
                    "2010-01-16",
                    "2010-01-24",
                    "2010-02-07",
                    "2011-01-08",
                    "2013-01-12",
                    "2014-01-12",
                    "2014-01-19",
                    "2014-02-02",
                    "2015-01-11",
                    "2016-01-17",
                    "2016-01-24",
                    "2016-02-07",
                ]
            ),
        }
    )
    superbowls = pd.DataFrame(
        {
            "event": "superbowl",
            "ds": pd.to_datetime(["2010-02-07", "2014-02-02", "2016-02-07"]),
        }
    )
    events_df = pd.concat((playoffs, superbowls))

    m = NeuralProphet(
        n_forecasts=1,
        loss_func="SmoothL1Loss",
        quantiles=[0.01, 0.99],
        epochs=epochs,
        batch_size=batch,
        learning_rate=LR,
        yearly_seasonality=season,
        weekly_seasonality=season,
        # daily_seasonality=False,
    )

    # add lagged regressors
    # # if m.n_lags > 0:
    #     df["A"] = df["y"].rolling(7, min_periods=1).mean()
    #     df["B"] = df["y"].rolling(30, min_periods=1).mean()
    #     m = m.add_lagged_regressor(name="A", n_lags=10)
    #     m = m.add_lagged_regressor(name="B", only_last_value=True)

    # add events
    m = m.add_events(["superbowl", "playoff"], lower_window=-1, upper_window=1, regularization=0.1)

    m = m.add_country_holidays("US", mode="additive", regularization=0.1)

    df["C"] = df["y"].rolling(7, min_periods=1).mean()
    df["D"] = df["y"].rolling(30, min_periods=1).mean()

    m = m.add_future_regressor(name="C", regularization=0.1)
    m = m.add_future_regressor(name="D", regularization=0.1)

    history_df = m.create_df_with_events(df, events_df)

    m.fit(history_df, freq="D")

    # periods = 90
    # regressors_future_df = pd.DataFrame(data={"C": df["C"][:periods], "D": df["D"][:periods]})
    # future_df = m.make_future_dataframe(
    #     df=history_df,
    #     regressors_df=regressors_future_df,
    #     events_df=events_df,
    #     periods=periods,
    #     n_historic_predictions=nrows,
    # )
    # m.predict(df=future_df)


def peyton_minus_events(nrows=NROWS, epochs=EPOCHS, batch=BATCH_SIZE, season=True):
    # log.info("testing: Uncertainty Estimation Peyton Manning")
    df = pd.read_csv(PEYTON_FILE, nrows=nrows)

    m = NeuralProphet(
        n_forecasts=1,
        loss_func="SmoothL1Loss",
        quantiles=[0.01, 0.99],
        epochs=epochs,
        batch_size=batch,
        learning_rate=LR,
        yearly_seasonality=season,
        weekly_seasonality=season,
        # daily_seasonality=False,
    )

    # add lagged regressors
    if m.n_lags > 0:
        df["A"] = df["y"].rolling(7, min_periods=1).mean()
        df["B"] = df["y"].rolling(30, min_periods=1).mean()
        m = m.add_lagged_regressor(name="A")
        m = m.add_lagged_regressor(name="B", only_last_value=True)

    df["C"] = df["y"].rolling(7, min_periods=1).mean()
    df["D"] = df["y"].rolling(30, min_periods=1).mean()

    m = m.add_future_regressor(name="C", regularization=0.1)
    m = m.add_future_regressor(name="D", regularization=0.1)

    history_df = df

    m.fit(history_df, freq="D")

    # periods = 90
    # regressors_future_df = pd.DataFrame(data={"C": df["C"][:periods], "D": df["D"][:periods]})
    # future_df = m.make_future_dataframe(
    #     df=history_df,
    #     regressors_df=regressors_future_df,
    #     periods=periods,
    #     n_historic_predictions=nrows,
    # )
    # m.predict(df=future_df)


def peyton_minus_regressors(nrows=NROWS, epochs=EPOCHS, batch=BATCH_SIZE, season=True):
    # log.info("testing: Uncertainty Estimation Peyton Manning")
    df = pd.read_csv(PEYTON_FILE, nrows=nrows)
    playoffs = pd.DataFrame(
        {
            "event": "playoff",
            "ds": pd.to_datetime(
                [
                    "2008-01-13",
                    "2009-01-03",
                    "2010-01-16",
                    "2010-01-24",
                    "2010-02-07",
                    "2011-01-08",
                    "2013-01-12",
                    "2014-01-12",
                    "2014-01-19",
                    "2014-02-02",
                    "2015-01-11",
                    "2016-01-17",
                    "2016-01-24",
                    "2016-02-07",
                ]
            ),
        }
    )
    superbowls = pd.DataFrame(
        {
            "event": "superbowl",
            "ds": pd.to_datetime(["2010-02-07", "2014-02-02", "2016-02-07"]),
        }
    )
    events_df = pd.concat((playoffs, superbowls))

    m = NeuralProphet(
        n_forecasts=1,
        loss_func="SmoothL1Loss",
        quantiles=[0.01, 0.99],
        epochs=epochs,
        batch_size=batch,
        learning_rate=LR,
        yearly_seasonality=season,
        weekly_seasonality=season,
        # daily_seasonality=False,
    )
    # add events
    m = m.add_events(["superbowl", "playoff"], lower_window=-1, upper_window=1, regularization=0.1)

    m = m.add_country_holidays("US", mode="additive", regularization=0.1)

    history_df = m.create_df_with_events(df, events_df)

    m.fit(history_df, freq="D")

    # periods = 90
    # future_df = m.make_future_dataframe(
    #     df=history_df,
    #     events_df=events_df,
    #     periods=periods,
    #     n_historic_predictions=nrows,
    # )
    # m.predict(df=future_df)


#######################################
# tic = time.perf_counter()
# test_uncertainty_estimation_yosemite_temps()
# toc = time.perf_counter()
# print(f"#### Time: {toc - tic:0.4f} for test_uncertainty_estimation_yosemite_temps")

# tic = time.perf_counter()
# test_uncertainty_estimation_peyton_manning()
# toc = time.perf_counter()
# print(f"#### Time: {toc - tic:0.4f} for test_uncertainty_estimation_peyton_manning")

# tic = time.perf_counter()
# test_uncertainty_estimation_air_travel()
# toc = time.perf_counter()
# print(f"#### Time: {toc - tic:0.4f} for test_uncertainty_estimation_air_travel")

# tic = time.perf_counter()
# test_uncertainty_estimation_multiple_quantiles()
# toc = time.perf_counter()
# print(f"#### Time: {toc - tic:0.4f} for test_uncertainty_estimation_multiple_quantiles")

# tic = time.perf_counter()
# test_split_conformal_prediction()
# toc = time.perf_counter()
# print(f"#### Time: {toc - tic:0.4f} for test_split_conformal_prediction")

# tic = time.perf_counter()
# test_asymmetrical_quantiles()
# toc = time.perf_counter()
# print(f"#### Time: {toc - tic:0.4f} for test_asymmetrical_quantiles")


# t0 = benchmark.Timer(
# stmt='test_uncertainty_estimation_yosemite_temps(x)',
# setup='from __main__ import test_uncertainty_estimation_yosemite_temps',
# globals={'x': x}
# )

# t1 = benchmark.Timer(
# stmt='test_uncertainty_estimation_peyton_manning(x)',
# setup='from __main__ import test_uncertainty_estimation_peyton_manning',
# # globals={'x': x}
# )

# print(t0.timeit(1))
# print(t1.timeit(1))


###############################


def measure_times():
    # Compare takes a list of measurements which we'll save in results.
    results = []

    epochs = [5]
    sizes = [100, 1000]
    # sizes = [100, 1000, 10000]
    batches = [128]
    seasons = [False, True]
    for ep, nrows, b, season in product(epochs, sizes, batches, seasons):
        # label and sub_label are the rows
        # description is the column
        label = "tests"
        sub_label = f"[rows: {nrows}, epochs:{ep}, batch:{b}, season:{season}]"
        for num_threads in [1]:  # [1, 4, 16, 64]
            results.append(
                benchmark.Timer(
                    stmt="yosemite(nrows, epochs, batch, season)",
                    setup="from __main__ import yosemite",
                    globals={"epochs": ep, "nrows": nrows, "batch": b, "season": season},
                    num_threads=num_threads,
                    label=label,
                    sub_label=sub_label,
                    description="yosemite",
                ).blocked_autorange(min_run_time=1)
            )
            results.append(
                benchmark.Timer(
                    stmt="peyton(nrows, epochs, batch, season)",
                    setup="from __main__ import peyton",
                    globals={"nrows": nrows, "epochs": ep, "batch": b, "season": season},
                    num_threads=num_threads,
                    label=label,
                    sub_label=sub_label,
                    description="peyton",
                ).blocked_autorange(min_run_time=1)
            )
            results.append(
                benchmark.Timer(
                    stmt="peyton_minus_events(nrows, epochs, batch, season)",
                    setup="from __main__ import peyton_minus_events",
                    globals={"nrows": nrows, "epochs": ep, "batch": b, "season": season},
                    num_threads=num_threads,
                    label=label,
                    sub_label=sub_label,
                    description="peyton_minus_events",
                ).blocked_autorange(min_run_time=1)
            )
            results.append(
                benchmark.Timer(
                    stmt="peyton_minus_regressors(nrows, epochs, batch, season)",
                    setup="from __main__ import peyton_minus_regressors",
                    globals={"nrows": nrows, "epochs": ep, "batch": b, "season": season},
                    num_threads=num_threads,
                    label=label,
                    sub_label=sub_label,
                    description="peyton_minus_regressors",
                ).blocked_autorange(min_run_time=1)
            )

    compare = benchmark.Compare(results)
    compare.print()


measure_times()
