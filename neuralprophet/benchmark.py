from dataclasses import dataclass, field
from typing import List, Generic, Optional, TypeVar, Tuple, Type
from abc import ABC, abstractmethod
import logging

import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet, df_utils
from prophet import Prophet

NeuralProphetModel = NeuralProphet
ProphetModel = Prophet

log = logging.getLogger("NP.benchmark")
log.warning("Benchmarking Framework is not covered by tests. Please report any bugs you find.")


def _calc_mae(
    predictions: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Calculates MAE error."""
    error_abs = np.abs(truth - predictions)
    return 1.0 * np.mean(error_abs)


def _calc_mse(
    predictions: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Calculates MSE error."""
    error_squared = np.square(truth - predictions)
    return 1.0 * np.mean(error_squared)


def _calc_rmse(
    predictions: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Calculates RMSE error."""
    mse = _calc_mse(predictions, truth)
    return np.sqrt(mse)


def _calc_mase(
    predictions: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Calculates MASE error.
        MASE = MAE / NaiveMAE,
    where: MAE = mean(|actual - forecast|)
    where: NaiveMAE = mean(|actual_[i] - actual_[i-1]|)
    """
    assert len(truth) > 1
    mae = _calc_mae(predictions, truth)
    naive_mae = _calc_mae(np.array(truth[:-1]), np.array(truth[1:]))
    return mae / (1e-9 + naive_mae)


def _calc_msse(
    predictions: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Calculates MSSE error.
    MSSE = MSE / NaiveMSE,
    where: MSE = mean((actual - forecast)^2)
    where: NaiveMSE = mean((actual_[i] - actual_[i-1])^2)
    """
    mse = _calc_mse(predictions, truth)
    naive_mse = _calc_mse(np.array(truth[:-1]), np.array(truth[1:]))
    return mse / (1e-9 + naive_mse)


def _calc_mape(
    predictions: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Calculates MAPE error."""
    error_relative = np.abs((truth - predictions) / truth)
    return 100.0 * np.mean(error_relative)


def _calc_smape(
    predictions: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Calculates SMAPE error."""
    error_relative_sym = np.abs(truth - predictions) / (np.abs(truth) + np.abs(predictions))
    return 100.0 * np.mean(error_relative_sym)


@dataclass
class Dataset:
    """
    example use:
    >>> dataset = Dataset(
    >>>     df = load_data('tmp-data', 'air_passengers.csv'),
    >>>     name = "air_passengers",
    >>>     freq = "MS",
    >>> ),
    """

    df: pd.DataFrame
    name: str
    freq: str


@dataclass
class Model(ABC):
    """
    example use:
    >>> models = []
    >>> for params in [{"n_changepoints": 5}, {"n_changepoints": 50},]:
    >>>     models.append(Model(
    >>>         params=params
    >>>         model_name="NeuralProphet",
    >>>         model_class=NeuralProphet,
    >>>     ))
    """

    params: dict
    model_name: str
    model_class: Type

    def __post_init__(self):
        self.model = self.model_class(**self.params)

    @abstractmethod
    def fit(self, df: pd.DataFrame, freq: str):
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame):
        pass

    def maybe_add_first_inputs_to_df(self, df_train, df_test):
        """
        if Model with lags: adds n_lags values to start of df_test.
        else (time-features only): returns unchanged df_test
        """
        return df_test.reset_index(drop=True)

    def maybe_drop_first_forecasts(self, predicted, df):
        """
        if Model with lags: removes firt n_lags values from predicted and df_test
        else (time-features only): returns unchanged df_test
        """
        return predicted.reset_index(drop=True), df.reset_index(drop=True)


@dataclass
class ProphetModel(Model):
    model_name: str = "Prophet"
    model_class: Type = Prophet

    def fit(self, df: pd.DataFrame, freq: str):
        self.freq = freq
        self.model = self.model.fit(df=df)

    def predict(self, df: pd.DataFrame):
        fcst = self.model.predict(df=df)
        fcst_df = pd.DataFrame({"time": fcst.ds, "yhat": fcst.yhat})
        return fcst_df


@dataclass
class NeuralProphetModel(Model):
    model_name: str = "NeuralProphet"
    model_class: Type = NeuralProphet

    def fit(self, df: pd.DataFrame, freq: str):
        self.freq = freq
        metrics = self.model.fit(df=df, freq=freq)

    def predict(self, df: pd.DataFrame):
        fcst = self.model.predict(df=df)
        if self.model.n_forecasts > 1:
            raise NotImplementedError
        fcst_df = pd.DataFrame({"time": fcst.ds, "yhat": fcst.yhat1})
        return fcst_df

    def maybe_add_first_inputs_to_df(self, df_train, df_test):
        """Adds last n_lags values from df_train to start of df_test."""
        if self.model.n_lags > 0:
            df_test = pd.concat([df_train.tail(self.model.n_lags), df_test], ignore_index=True)
        return df_test.reset_index(drop=True)

    def maybe_drop_first_forecasts(self, predicted, df):
        """
        if Model with lags: removes firt n_lags values from predicted and df
        else (time-features only): returns unchanged df
        """
        if self.model.n_lags > 0:
            predicted = predicted[self.model.n_lags :]
            df = df[self.model.n_lags :]
        return predicted.reset_index(drop=True), df.reset_index(drop=True)


@dataclass
class Experiment(ABC):
    model_class: Model
    params: dict
    data: Dataset
    metrics: List[str]
    test_percentage: float
    experiment_name: dict = field(init=False)

    def __post_init__(self):
        self.experiment_name = {
            "data": self.data.name,
            "model": self.model_class.model_name,
            "params": str(self.params),
        }

        self.error_funcs = {
            "MAE": _calc_mae,
            "MSE": _calc_mse,
            "RMSE": _calc_rmse,
            "MASE": _calc_mase,
            "MSSE": _calc_msse,
            "MAPE": _calc_mape,
            "SMAPE": _calc_smape,
        }

    def _evaluate_model(self, model, df_train, df_test):
        df_test = model.maybe_add_first_inputs_to_df(df_train, df_test)
        fcst_train = model.predict(df_train)
        fcst_test = model.predict(df_test)
        fcst_train, df_train = model.maybe_drop_first_forecasts(fcst_train, df_train)
        fcst_test, df_test = model.maybe_drop_first_forecasts(fcst_test, df_test)

        result_train = self.experiment_name.copy()
        result_test = self.experiment_name.copy()
        for metric in self.metrics:
            # todo: parallelize
            result_train[metric] = self.error_funcs[metric](fcst_train["yhat"], df_train["y"])
            result_test[metric] = self.error_funcs[metric](fcst_test["yhat"], df_test["y"])
        return result_train, result_test

    def get_metric(self, predictions, truth, metric):
        """Get benchmark metric

        Args:
            predictions (pd.Series): series of forecasted values
            truth (pd.Series): series of true values
            metric (str): name of metric
        Returns:
            error_values (dict): errors stored in a dict
        """
        error_value = self.error_funcs[metric](predictions, truth)
        return error_value

    @abstractmethod
    def run(self):
        pass


@dataclass
class SimpleExperiment(Experiment):
    """
    use example:
    >>> ts = Dataset(df = air_passengers_df, name = "air_passengers", freq = "MS")
    >>> params = {"seasonality_mode": "multiplicative"}
    >>> exp = SimpleExperiment(
    >>>     model_class=NeuralProphetModel,
    >>>     params=params,
    >>>     data=ts,
    >>>     metrics=["MAE", "MSE"],
    >>>     test_percentage=25,
    >>> )
    >>> result_train, result_val = exp.run()
    """

    def run(self):
        df_train, df_test = df_utils.split_df(
            df=self.data.df,
            n_lags=0,
            n_forecasts=1,
            valid_p=self.test_percentage / 100.0,
        )
        model = self.model_class(self.params)
        model.fit(df=df_train, freq=self.data.freq)
        result_train, result_test = self._evaluate_model(model, df_train, df_test)
        return result_train, result_test


@dataclass
class CrossValidationExperiment(Experiment):
    """
    >>> ts = Dataset(df = air_passengers_df, name = "air_passengers", freq = "MS")
    >>> params = {"seasonality_mode": "multiplicative"}
    >>> exp = CrossValidationExperiment(
    >>>     model_class=NeuralProphetModel,
    >>>     params=params,
    >>>     data=ts,
    >>>     metrics=["MAE", "MSE"],
    >>>     test_percentage=10,
    >>>     num_folds=3,
    >>>     fold_overlap_pct=0,
    >>> )
    >>> result_train, result_train, result_val = exp.run()
    """

    num_folds: int
    fold_overlap_pct: float = 0

    def run(self):
        folds = df_utils.crossvalidation_split_df(
            df=self.data.df,
            n_lags=0,
            n_forecasts=1,
            k=self.num_folds,
            fold_pct=self.test_percentage / 100.0,
            fold_overlap_pct=self.fold_overlap_pct / 100.0,
        )
        # init empty dicts with list for fold-wise metrics
        results_cv_train = self.experiment_name.copy()
        results_cv_test = self.experiment_name.copy()
        for m in self.metrics:
            results_cv_train[m] = []
            results_cv_test[m] = []
        for df_train, df_test in folds:
            # todo: parallelize
            model = self.model_class(self.params)
            model.fit(df=df_train, freq=self.data.freq)
            result_train, result_test = self._evaluate_model(model, df_train, df_test)
            for m in self.metrics:
                results_cv_train[m].append(result_train[m])
                results_cv_test[m].append(result_test[m])
        return results_cv_train, results_cv_test


@dataclass
class SimpleBenchmark:
    """
    use example:
    >>> benchmark = SimpleBenchmark(
    >>>     model_classes_and_params=model_classes_and_params, # iterate over this list of tuples
    >>>     datasets=dataset_list, # iterate over this list
    >>>     metrics=["MAE", "MSE"],
    >>>     test_percentage=25,
    >>> )
    >>> results_train, results_val = benchmark.run()
    """

    model_classes_and_params: List[Tuple[Model, dict]]
    datasets: List[Dataset]
    metrics: List[str]
    test_percentage: float

    def setup_experiments(self):
        experiments = []
        for ts in self.datasets:
            for model_class, params in self.model_classes_and_params:
                exp = SimpleExperiment(
                    model_class=model_class,
                    params=params,
                    data=ts,
                    metrics=self.metrics,
                    test_percentage=self.test_percentage,
                )
                experiments.append(exp)
        return experiments

    def run(self):
        experiments = self.setup_experiments()
        # setup DataFrame to store each experiment in a row
        cols = list(experiments[0].experiment_name.keys()) + self.metrics
        df_metrics_train = pd.DataFrame(columns=cols)
        df_metrics_test = pd.DataFrame(columns=cols)
        for exp in experiments:
            # todo: parallelize
            exp.metrics = self.metrics
            res_train, res_test = exp.run()
            df_metrics_train = df_metrics_train.append(res_train, ignore_index=True)
            df_metrics_test = df_metrics_test.append(res_test, ignore_index=True)
        return df_metrics_train, df_metrics_test


@dataclass
class CrossValidationBenchmark(SimpleBenchmark):
    """
    example use:
    >>> benchmark_cv = CrossValidationBenchmark(
    >>>     model_classes_and_params=model_classes_and_params, # iterate over this list of tuples
    >>>     datasets=dataset_list, # iterate over this list
    >>>     metrics=["MAE", "MSE"],
    >>>     test_percentage=10,
    >>>     num_folds=3,
    >>>     fold_overlap_pct=0,
    >>> )
    >>> results_summary, results_train, results_val = benchmark_cv.run()
    """

    num_folds: int
    fold_overlap_pct: float = 0

    def setup_experiments(self):
        experiments = []
        for ts in self.datasets:
            for model_class, params in self.model_classes_and_params:
                exp = CrossValidationExperiment(
                    model_class=model_class,
                    params=params,
                    data=ts,
                    metrics=self.metrics,
                    test_percentage=self.test_percentage,
                    num_folds=self.num_folds,
                    fold_overlap_pct=self.fold_overlap_pct,
                )
                experiments.append(exp)
        return experiments

    def run(self):
        df_metrics_train, df_metrics_test = super().run()
        df_metrics_summary_train = summarize_cv_metrics(df_metrics_train, self.metrics)
        df_metrics_summary_train["split"] = "train"
        df_metrics_summary_test = summarize_cv_metrics(df_metrics_test, self.metrics)
        df_metrics_summary_test["split"] = "test"
        df_metrics_summary = df_metrics_summary_train.append(df_metrics_summary_test)
        return df_metrics_summary, df_metrics_train, df_metrics_test


def summarize_cv_metrics(df_metrics, metrics, name=None):
    df_metrics_summary = df_metrics.copy(deep=True)
    name = "" if name is None else "_{}".format(name)
    for metric in metrics:
        df_metrics_summary[metric + name] = df_metrics[metric].copy(deep=True).apply(lambda x: np.array(x).mean())
        df_metrics_summary[metric + "_std" + name] = (
            df_metrics[metric].copy(deep=True).apply(lambda x: np.array(x).std())
        )
    return df_metrics_summary


def debug_experiment():
    import os
    import pathlib

    DIR = pathlib.Path(__file__).parent.parent.absolute()
    DATA_DIR = os.path.join(DIR, "tests", "test-data")
    AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
    air_passengers_df = pd.read_csv(AIR_FILE)

    ts = Dataset(df=air_passengers_df, name="air_passengers", freq="MS")
    params = {
        "seasonality_mode": "multiplicative",
    }
    exp = SimpleExperiment(
        model_class=NeuralProphetModel,
        params=params,
        data=ts,
        metrics=["MAE", "MSE", "RMSE", "MASE", "MSSE"],
        test_percentage=25,
    )
    result_train, result_val = exp.run()
    print(result_val)

    ts = Dataset(df=air_passengers_df, name="air_passengers", freq="MS")
    params = {
        "seasonality_mode": "multiplicative",
    }
    exp_cv = CrossValidationExperiment(
        model_class=ProphetModel,
        params=params,
        data=ts,
        metrics=["MAE", "MSE", "RMSE", "MASE", "MSSE", "MAPE", "SMAPE"],
        test_percentage=10,
        num_folds=3,
        fold_overlap_pct=0,
    )
    result_train, result_val = exp_cv.run()
    print(result_val)


def debug_benchmark():
    import os
    import pathlib

    DIR = pathlib.Path(__file__).parent.parent.absolute()
    DATA_DIR = os.path.join(DIR, "tests", "test-data")
    PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
    AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
    YOS_FILE = os.path.join(DATA_DIR, "yosemite_temps.csv")
    air_passengers_df = pd.read_csv(AIR_FILE)
    peyton_manning_df = pd.read_csv(PEYTON_FILE)
    dataset_list = [
        Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
        # Dataset(df=peyton_manning_df, name="peyton_manning", freq="D"),
        # Dataset(df = retail_sales_df, name = "retail_sales", freq = "D"),
        # Dataset(df = yosemite_temps_df, name = "yosemite_temps", freq = "5min"),
        # Dataset(df = ercot_load_df, name = "ercot_load", freq = "H"),
    ]
    model_classes_and_params = [
        (NeuralProphetModel, {"seasonality_mode": "multiplicative", "learning_rate": 0.1}),
        (ProphetModel, {"seasonality_mode": "multiplicative"}),
        # (NeuralProphetModel, {"learning_rate": 0.1}),
        # (ProphetModel, {}),
        # (NeuralProphetModel, {"seasonality_mode": "multiplicative", "learning_rate": 0.1}),
    ]
    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,  # iterate over this list of tuples
        datasets=dataset_list,  # iterate over this list
        metrics=["MAE", "MSE", "RMSE", "MASE", "MSSE", "MAPE", "SMAPE"],
        test_percentage=25,
    )
    results_train, results_test = benchmark.run()
    print(results_test.to_string())

    benchmark_cv = CrossValidationBenchmark(
        model_classes_and_params=model_classes_and_params,  # iterate over this list of tuples
        datasets=dataset_list,  # iterate over this list
        metrics=["MAE", "MSE", "RMSE", "MASE", "MSSE", "MAPE", "SMAPE"],
        test_percentage=10,
        num_folds=3,
        fold_overlap_pct=0,
    )
    results_summary, results_train, results_test = benchmark_cv.run()
    print(results_summary.to_string())
    print(results_train.to_string())
    print(results_test.to_string())


if __name__ == "__main__":
    debug_experiment()
    debug_benchmark()
