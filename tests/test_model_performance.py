#!/usr/bin/env python3

import json
import logging
import os
import pathlib
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from plotly_resampler import unregister_plotly_resampler

from neuralprophet import NeuralProphet, set_random_seed

log = logging.getLogger("NP.test")
log.setLevel("DEBUG")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
YOS_FILE = os.path.join(DATA_DIR, "yosemite_temps.csv")
ENERGY_PRICE_DAILY_FILE = os.path.join(DATA_DIR, "tutorial04_kaggle_energy_daily_temperature.csv")

# Important to set seed for reproducibility
set_random_seed(42)


def get_system_speed():
    repeats = 5
    benchmarks = np.array([])
    for a in range(0, repeats):
        start = time.time()
        for i in range(0, 1000):
            for x in range(1, 1000):
                3.141592 * 2**x
            for x in range(1, 1000):
                float(x) / 3.141592
            for x in range(1, 1000):
                float(3.141592) / x

        end = time.time()
        duration = end - start
        duration = round(duration, 3)
        benchmarks = np.append(benchmarks, duration)

    log.info(f"System speed: {round(np.mean(benchmarks), 5)}s")
    log.info(f"Standart deviation: {round(np.std(benchmarks), 5)}s")
    return benchmarks.mean(), benchmarks.std()


def create_metrics_plot(metrics):
    # Deactivate the resampler since it is not compatible with kaleido (image export)
    unregister_plotly_resampler()

    # Plotly params
    prediction_color = "#2d92ff"
    actual_color = "black"
    line_width = 2
    xaxis_args = {"showline": True, "mirror": True, "linewidth": 1.5, "showgrid": False}
    yaxis_args = {
        "showline": True,
        "mirror": True,
        "linewidth": 1.5,
        "showgrid": False,
        "rangemode": "tozero",
        "type": "log",
    }
    layout_args = {
        "autosize": True,
        "template": "plotly_white",
        "margin": go.layout.Margin(l=0, r=10, b=0, t=30, pad=0),
        "font": dict(size=10),
        "title": dict(font=dict(size=10)),
        "width": 1000,
        "height": 200,
    }

    metric_cols = [col for col in metrics.columns if not ("_val" in col or col == "RegLoss" or col == "epoch")]
    fig = make_subplots(rows=1, cols=len(metric_cols), subplot_titles=metric_cols)
    for i, metric in enumerate(metric_cols):
        fig.add_trace(
            go.Scatter(
                y=metrics[metric],
                name=metric,
                mode="lines",
                line=dict(color=prediction_color, width=line_width),
                legendgroup=metric,
            ),
            row=1,
            col=i + 1,
        )
        if f"{metric}_val" in metrics.columns:
            fig.add_trace(
                go.Scatter(
                    y=metrics[f"{metric}_val"],
                    name=f"{metric}_val",
                    mode="lines",
                    line=dict(color=actual_color, width=line_width),
                    legendgroup=metric,
                ),
                row=1,
                col=i + 1,
            )
        if metric == "Loss":
            fig.add_trace(
                go.Scatter(
                    y=metrics["RegLoss"],
                    name="RegLoss",
                    mode="lines",
                    line=dict(color=actual_color, width=line_width),
                    legendgroup=metric,
                ),
                row=1,
                col=i + 1,
            )
    fig.update_xaxes(xaxis_args)
    fig.update_yaxes(yaxis_args)
    fig.update_layout(layout_args)
    return fig


def test_PeytonManning():
    df = pd.read_csv(PEYTON_FILE)
    m = NeuralProphet(
        # learning_rate=0.01,
        # epochs=3,
    )
    df_train, df_test = m.split_df(df=df, freq="D", valid_p=0.1)

    system_speed, std = get_system_speed()
    start = time.time()
    metrics = m.fit(df_train, validation_df=df_test, freq="D")  # , early_stopping=True)
    end = time.time()

    accuracy_metrics = metrics.to_dict("records")[-1]
    accuracy_metrics["time"] = round(end - start, 2)
    accuracy_metrics["system_performance"] = round(system_speed, 5)
    accuracy_metrics["system_std"] = round(std, 5)
    with open(os.path.join(DIR, "tests", "metrics", "PeytonManning.json"), "w") as outfile:
        json.dump(accuracy_metrics, outfile)

    create_metrics_plot(metrics).write_image(os.path.join(DIR, "tests", "metrics", "PeytonManning.svg"))


def test_YosemiteTemps():
    df = pd.read_csv(YOS_FILE)
    m = NeuralProphet(
        # learning_rate=0.01,
        # epochs=3,
        n_lags=36,
        n_forecasts=12,
        changepoints_range=0.9,
        n_changepoints=30,
        weekly_seasonality=False,
    )
    df_train, df_test = m.split_df(df=df, freq="5min", valid_p=0.1)

    system_speed, std = get_system_speed()
    start = time.time()
    metrics = m.fit(df_train, validation_df=df_test, freq="5min")  # , early_stopping=True)
    end = time.time()

    accuracy_metrics = metrics.to_dict("records")[-1]
    accuracy_metrics["time"] = round(end - start, 2)
    accuracy_metrics["system_performance"] = round(system_speed, 5)
    accuracy_metrics["system_std"] = round(std, 5)
    with open(os.path.join(DIR, "tests", "metrics", "YosemiteTemps.json"), "w") as outfile:
        json.dump(accuracy_metrics, outfile)

    create_metrics_plot(metrics).write_image(os.path.join(DIR, "tests", "metrics", "YosemiteTemps.svg"))


def test_AirPassengers():
    df = pd.read_csv(AIR_FILE)
    m = NeuralProphet(
        # learning_rate=0.01,
        # epochs=3,
        seasonality_mode="multiplicative",
    )
    df_train, df_test = m.split_df(df=df, freq="MS", valid_p=0.1)

    system_speed, std = get_system_speed()
    start = time.time()
    metrics = m.fit(df_train, validation_df=df_test, freq="MS")  # , early_stopping=True)
    end = time.time()

    accuracy_metrics = metrics.to_dict("records")[-1]
    accuracy_metrics["time"] = round(end - start, 2)
    accuracy_metrics["system_performance"] = round(system_speed, 5)
    accuracy_metrics["system_std"] = round(std, 5)
    with open(os.path.join(DIR, "tests", "metrics", "AirPassengers.json"), "w") as outfile:
        json.dump(accuracy_metrics, outfile)

    create_metrics_plot(metrics).write_image(os.path.join(DIR, "tests", "metrics", "AirPassengers.svg"))


def test_EnergyPriceDaily():
    df = pd.read_csv(ENERGY_PRICE_DAILY_FILE)
    df["temp"] = df["temperature"]

    m = NeuralProphet(
        # learning_rate=0.01,
        # epochs=3,
        n_forecasts=7,
        n_changepoints=0,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        n_lags=14,
    )
    m.add_lagged_regressor("temp", n_lags=3)
    m.add_future_regressor("temperature")

    df_train, df_test = m.split_df(df=df, freq="D", valid_p=0.1)

    system_speed, std = get_system_speed()
    start = time.time()
    metrics = m.fit(df_train, validation_df=df_test, freq="D")  # , early_stopping=True)
    end = time.time()

    accuracy_metrics = metrics.to_dict("records")[-1]
    accuracy_metrics["time"] = round(end - start, 2)
    accuracy_metrics["system_performance"] = round(system_speed, 5)
    accuracy_metrics["system_std"] = round(std, 5)
    with open(os.path.join(DIR, "tests", "metrics", "EnergyPriceDaily.json"), "w") as outfile:
        json.dump(accuracy_metrics, outfile)

    create_metrics_plot(metrics).write_image(os.path.join(DIR, "tests", "metrics", "EnergyPriceDaily.svg"))


def test_EnergyDailyDeep():
    ### Temporary Test for on-the-fly sampling - very time consuming!

    df = pd.read_csv(ENERGY_PRICE_DAILY_FILE)
    df = df[df["ds"] < "2018-01-01"]
    df["temp"] = df["temperature"]
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["ID"] = "test"

    # Conditional Seasonality
    df["winter"] = np.where(
        df["ds"].dt.month.isin(
            [
                10,
                11,
                12,
                1,
                2,
                3,
            ]
        ),
        1,
        0,
    )
    df["summer"] = np.where(df["ds"].dt.month.isin([4, 5, 6, 7, 8, 9]), 1, 0)
    df["winter"] = pd.to_numeric(df["winter"], errors="coerce")
    df["summer"] = pd.to_numeric(df["summer"], errors="coerce")

    # Normalize Temperature
    df["temp"] = (df["temp"] - 65.0) / 50.0

    # df
    df = df[["ID", "ds", "y", "temp", "winter", "summer"]]

    # Hyperparameter
    tuned_params = {
        "n_lags": 15,
        "newer_samples_weight": 2.0,
        "n_changepoints": 0,
        "yearly_seasonality": 10,
        "weekly_seasonality": False,  # due to conditional daily seasonality
        "daily_seasonality": False,  # due to data freq
        "batch_size": 64,
        "ar_layers": [16, 32, 16, 8],
        "lagged_reg_layers": [32, 16],
        # not tuned
        "n_forecasts": 7,
        "learning_rate": 0.001,
        "epochs": 30,
        "trend_global_local": "global",
        "season_global_local": "global",
        "drop_missing": True,
        "normalize": "standardize",
    }

    # Uncertainty Quantification
    confidence_lv = 0.98
    quantile_list = [round(((1 - confidence_lv) / 2), 2), round((confidence_lv + (1 - confidence_lv) / 2), 2)]

    # Check if GPU is available
    use_gpu = torch.cuda.is_available()

    # Set trainer configuration
    trainer_configs = {
        "accelerator": "gpu" if use_gpu else "cpu",
    }
    print(f"Using {'GPU' if use_gpu else 'CPU'}")

    # Model
    m = NeuralProphet(**tuned_params, **trainer_configs, quantiles=quantile_list)

    # Lagged Regressor
    m.add_lagged_regressor(names="temp", n_lags=7, normalize="standardize")

    # Conditional Seasonality
    m.add_seasonality(name="winter", period=7, fourier_order=6, condition_name="winter")
    m.add_seasonality(name="summer", period=7, fourier_order=6, condition_name="summer")

    # Holidays
    m.add_country_holidays(country_name="US", lower_window=-1, upper_window=1)

    # Split
    df_train = df[df["ds"] < "2016-05-01"]
    df_test = df[df["ds"] >= "2016-05-01"]

    # Training & Predict
    _ = m.fit(df=df_train, freq="D", num_workers=4)
    _ = m.predict(df_test)


# TODO: adapt to hourly dataset with multiple IDs
def test_EnergyHourlyDeep():
    ### Temporary Test for on-the-fly sampling - very time consuming!

    df = pd.read_csv(ENERGY_PRICE_DAILY_FILE)
    df["temp"] = df["temperature"]
    df = df.drop(columns="temperature")
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = df.drop("ds", axis=1)
    df["ds"] = pd.date_range(start="2015-01-01 00:00:00", periods=len(df), freq="H")
    df["ID"] = "test"

    df_id = df[["ds", "y", "temp"]].copy()
    df_id["ID"] = "test2"
    df_id["y"] = df_id["y"] * 0.3
    df_id["temp"] = df_id["temp"] * 0.4
    df = pd.concat([df, df_id], ignore_index=True)

    # Conditional Seasonality
    df["winter"] = np.where(
        df["ds"].dt.month.isin([1]),
        1,
        0,
    )
    df["summer"] = np.where(df["ds"].dt.month.isin([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), 1, 0)
    df["winter"] = pd.to_numeric(df["winter"], errors="coerce")
    df["summer"] = pd.to_numeric(df["summer"], errors="coerce")

    # Normalize Temperature
    df["temp"] = (df["temp"] - 65.0) / 50.0

    # df
    df = df[["ID", "ds", "y", "temp", "winter", "summer"]]

    # Hyperparameter
    tuned_params = {
        "n_lags": 24 * 15,
        "newer_samples_weight": 2.0,
        "n_changepoints": 0,
        "yearly_seasonality": 10,
        "weekly_seasonality": True,
        "daily_seasonality": False,  # due to conditional daily seasonality
        "batch_size": 128,
        "ar_layers": [32, 64, 32, 16],
        "lagged_reg_layers": [32, 32],
        # not tuned
        "n_forecasts": 33,
        "learning_rate": 0.001,
        "epochs": 30,
        "trend_global_local": "global",
        "season_global_local": "global",
        "drop_missing": True,
        "normalize": "standardize",
    }

    # Uncertainty Quantification
    confidence_lv = 0.98
    quantile_list = [round(((1 - confidence_lv) / 2), 2), round((confidence_lv + (1 - confidence_lv) / 2), 2)]

    # Check if GPU is available
    use_gpu = torch.cuda.is_available()

    # Set trainer configuration
    trainer_configs = {
        "accelerator": "gpu" if use_gpu else "cpu",
    }
    print(f"Using {'GPU' if use_gpu else 'CPU'}")

    # Model
    m = NeuralProphet(**tuned_params, **trainer_configs, quantiles=quantile_list)

    # Lagged Regressor
    m.add_lagged_regressor(names="temp", n_lags=33, normalize="standardize")

    # Conditional Seasonality
    m.add_seasonality(name="winter", period=1, fourier_order=6, condition_name="winter")
    m.add_seasonality(name="summer", period=1, fourier_order=6, condition_name="summer")

    # Holidays
    m.add_country_holidays(country_name="US", lower_window=-1, upper_window=1)

    # Split
    df_train = df[df["ds"] < "2015-03-01"]
    df_test = df[df["ds"] >= "2015-03-01"]

    # Training & Predict
    _ = m.fit(df=df_train, freq="H", num_workers=4, early_stopping=True)
    _ = m.predict(df_test)
