import logging
import os
import pathlib
import time
from itertools import product

import pandas as pd
import pytest
import torch.utils.benchmark as benchmark

from neuralprophet import NeuralProphet, uncertainty_evaluate

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
NROWS = 256
EPOCHS = 10
BATCH_SIZE = 128
LR = 1.0


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


############################33333
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
