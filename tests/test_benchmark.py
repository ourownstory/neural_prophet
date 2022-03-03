#!/usr/bin/env python3

import pytest
import os
import pathlib
import pandas as pd
import logging
import matplotlib.pyplot as plt

from neuralprophet.benchmark import Dataset, NeuralProphetModel, ProphetModel
from neuralprophet.benchmark import SimpleBenchmark, CrossValidationBenchmark
from neuralprophet.benchmark import SimpleExperiment, CrossValidationExperiment
from neuralprophet.benchmark import ManualBenchmark, ManualCVBenchmark

log = logging.getLogger("NP.test")
log.setLevel("WARNING")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
NROWS = 256
EPOCHS = 2
BATCH_SIZE = 64
LR = 1.0

PLOT = False


def test_benchmark_simple():
    peyton_manning_df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)

    dataset_list = [
        Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
        Dataset(df=peyton_manning_df, name="peyton_manning", freq="D"),
    ]
    model_classes_and_params = [
        (NeuralProphetModel, {"epochs": EPOCHS}),
        (NeuralProphetModel, {"seasonality_mode": "multiplicative", "learning_rate": 0.1, "epochs": EPOCHS}),
        # (ProphetModel, {"seasonality_mode": "multiplicative"}) # needs to be installed
    ]
    log.debug("{}".format(model_classes_and_params))

    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,  # iterate over this list of tuples
        datasets=dataset_list,  # iterate over this list
        metrics=["MAE", "MSE", "MASE", "RMSE"],
        test_percentage=25,
    )
    results_train, results_test = benchmark.run()

    log.debug("{}".format(results_test))


def test_benchmark_CV():
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)
    peyton_manning_df = pd.read_csv(PEYTON_FILE, nrows=NROWS)
    dataset_list = [
        Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
        Dataset(df=peyton_manning_df, name="peyton_manning", freq="D"),
    ]
    model_classes_and_params = [
        (NeuralProphetModel, {"seasonality_mode": "multiplicative", "learning_rate": 0.1, "epochs": EPOCHS}),
        # (ProphetModel, {"seasonality_mode": "multiplicative"}) # needs to be installed
    ]
    log.debug("{}".format(model_classes_and_params))

    benchmark_cv = CrossValidationBenchmark(
        model_classes_and_params=model_classes_and_params,  # iterate over this list of tuples
        datasets=dataset_list,  # iterate over this list
        metrics=["MASE", "RMSE"],
        test_percentage=10,
        num_folds=3,
        fold_overlap_pct=0,
    )
    results_summary, results_train, results_test = benchmark_cv.run()
    log.debug("{}".format(results_summary))
    if PLOT:
        # model plot
        # air_passengers = results_summary[results_summary['data'] == 'air_passengers']
        # air_passengers = air_passengers[air_passengers['split'] == 'test']
        # plt_air = air_passengers.plot(x='model', y='RMSE', kind='barh')
        # data plot
        air_passengers = results_summary[results_summary["split"] == "test"]
        plt_air = air_passengers.plot(x="data", y="MASE", kind="barh")
        plt.show()


def test_benchmark_manual():
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)
    metrics = ["MAE", "MSE", "RMSE", "MASE", "RMSSE", "MAPE", "SMAPE"]
    experiments = [
        SimpleExperiment(
            model_class=NeuralProphetModel,
            params={"epochs": EPOCHS},
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=25,
        ),
        SimpleExperiment(
            model_class=NeuralProphetModel,
            params={"seasonality_mode": "multiplicative", "learning_rate": 0.1, "epochs": EPOCHS},
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=25,
        ),
        # needs to be installed
        # SimpleExperiment(
        #     model_class=ProphetModel,
        #     params={"seasonality_mode": "multiplicative", },
        #     data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
        #     metrics=metrics,
        #     test_percentage=25,
        # ),
    ]
    benchmark = ManualBenchmark(
        experiments=experiments,
        metrics=metrics,
    )
    results_train, results_test = benchmark.run()
    log.debug("{}".format(results_test))


def test_benchmark_manualCV():
    air_passengers_df = pd.read_csv(AIR_FILE, nrows=NROWS)
    metrics = ["MAE", "MSE", "RMSE", "MASE", "RMSSE", "MAPE", "SMAPE"]
    experiments = [
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={"epochs": EPOCHS},
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=10,
            num_folds=3,
            fold_overlap_pct=0,
        ),
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={"epochs": EPOCHS, "seasonality_mode": "multiplicative", "learning_rate": 0.1},
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=10,
            num_folds=3,
            fold_overlap_pct=0,
        ),
        # needs to be installed
        # CrossValidationExperiment(
        #     model_class=ProphetModel,
        #     params={"seasonality_mode": "multiplicative", },
        #     data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
        #     metrics=metrics,
        #     test_percentage=10,
        #     num_folds=3,
        #     fold_overlap_pct=0,
        # ),
    ]
    benchmark_cv = ManualCVBenchmark(
        experiments=experiments,
        metrics=metrics,
    )
    results_summary, results_train, results_test = benchmark_cv.run()
    log.debug("{}".format(results_summary))
