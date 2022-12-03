#!/usr/bin/env python3

import json
import logging
import os
import pathlib
import time

import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from neuralprophet import NeuralProphet, set_random_seed

log = logging.getLogger("NP.test")
log.setLevel("DEBUG")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
YOS_FILE = os.path.join(DATA_DIR, "yosemite_temps.csv")

# Important to set seed for reproducibility
set_random_seed(42)


def create_metrics_plot(metrics):
    # Plotly params
    prediction_color = "#2d92ff"
    actual_color = "black"
    line_width = 2
    marker_size = 4
    xaxis_args = {"showline": True, "mirror": True, "linewidth": 1.5, "showgrid": False}
    yaxis_args = {"showline": True, "mirror": True, "linewidth": 1.5, "showgrid": False, "rangemode": "tozero"}
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
    m = NeuralProphet(early_stopping=True)
    df_train, df_test = m.split_df(df=df, freq="D", valid_p=0.2)
    start = time.time()
    metrics = m.fit(df_train, validation_df=df_test, freq="D")
    end = time.time()

    accuracy_metrics = metrics.to_dict("records")[-1]
    accuracy_metrics["time"] = round(end - start, 2)
    with open(os.path.join(DIR, "tests", "metrics", "PeytonManning.json"), "w") as outfile:
        json.dump(accuracy_metrics, outfile)

    create_metrics_plot(metrics).write_image(os.path.join(DIR, "tests", "metrics", "PeytonManning.svg"))


def test_YosemiteTemps():
    df = pd.read_csv(YOS_FILE)
    m = NeuralProphet(
        n_lags=24,
        n_forecasts=24,
        changepoints_range=0.95,
        n_changepoints=30,
        weekly_seasonality=False,
        early_stopping=True,
    )
    df_train, df_test = m.split_df(df=df, freq="5min", valid_p=0.2)
    start = time.time()
    metrics = m.fit(df_train, validation_df=df_test, freq="5min")
    end = time.time()

    accuracy_metrics = metrics.to_dict("records")[-1]
    accuracy_metrics["time"] = round(end - start, 2)
    with open(os.path.join(DIR, "tests", "metrics", "YosemiteTemps.json"), "w") as outfile:
        json.dump(accuracy_metrics, outfile)

    create_metrics_plot(metrics).write_image(os.path.join(DIR, "tests", "metrics", "YosemiteTemps.svg"))


def test_AirPassengers():
    df = pd.read_csv(AIR_FILE)
    m = NeuralProphet(seasonality_mode="multiplicative", early_stopping=True)
    df_train, df_test = m.split_df(df=df, freq="MS", valid_p=0.2)
    start = time.time()
    metrics = m.fit(df_train, validation_df=df_test, freq="MS")
    end = time.time()

    accuracy_metrics = metrics.to_dict("records")[-1]
    accuracy_metrics["time"] = round(end - start, 2)
    with open(os.path.join(DIR, "tests", "metrics", "AirPassengers.json"), "w") as outfile:
        json.dump(accuracy_metrics, outfile)

    create_metrics_plot(metrics).write_image(os.path.join(DIR, "tests", "metrics", "AirPassengers.svg"))
