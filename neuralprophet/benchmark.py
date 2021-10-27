from dataclasses import dataclass, field
from typing import List, Generic, Optional, TypeVar, Tuple, Type
from abc import ABC, abstractmethod
import logging

import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet, df_utils
from fbprophet import Prophet

NeuralProphetModel = NeuralProphet
ProphetModel = Prophet

log = logging.getLogger("NP.benchmark")
log.warning("Benchmarking Framework is not covered by tests. Please report any bugs you find.")


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


@dataclass
class NeuralProphetModel(Model):
    model_name: str = "NeuralProphet"
    model_class: Type = NeuralProphet

    def maybe_add_first_inputs_to_df(self, df_train, df_val):
        """
        if NeuralProphetModel: add last n_lags values to start of df_val.
        else (ProphetModel): -
        """
        df_val = pd.concat([df_train.tail(self.model.n_lags), df_val], ignore_index=True)
        return df_val

    def fit(self, df: pd.DataFrame, freq: str):
        self.freq = freq
        metrics = self.model.fit(df=df, freq=freq)

    def predict(self, df: pd.DataFrame):
        fcst = self.model.predict(df=df)
        if self.model.n_forecasts > 1:
            raise NotImplementedError
        self.fcst_df = pd.DataFrame({"time": fcst.ds, "fcst": fcst.yhat1})
        return self.fcst_df


@dataclass
class ProphetModel(Model):
    model_name: str = "Prophet"
    model_class: Type = Prophet

    def maybe_add_first_inputs_to_df(self, df_train, df_val):
        """
        if NeuralProphetModel: adds n_lags values to start of df_val.
        else (ProphetModel): -
        """
        return df_val

    def fit(self, df: pd.DataFrame, freq: str):
        self.freq = freq
        self.model = self.model.fit(df=df)
        return None

    def predict(self, df: pd.DataFrame):
        steps = len(df)
        future = self.model.make_future_dataframe(periods=steps, freq=self.freq, include_history=False)
        fcst = self.model.predict(df=future)
        self.fcst_df = pd.DataFrame({"time": fcst.ds, "fcst": fcst.yhat})
        return self.fcst_df


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
            "MAPE": _calc_mape,
            "SMAPE": _calc_smape,
        }

    def get_metric(self, truth, predictions, metric):
        """Get benchmark metric

        Args:
            y (pd.Series): series of labels
            fcst (pd.Series): series of forecasts
            metric (str): name of metric
        Returns:
            error_values (dict): errors stored in a dict
        """
        error_value = self.error_funcs[metric](self, truth, predictions)
        return error_value

    @abstractmethod
    def run(self):
        pass


def _calc_mae(
    self,
    predictions: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Calculates MAE error."""
    diffs = np.abs(truth - predictions)
    return diffs.mean()


def _calc_mse(
    self,
    predictions: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Calculates MSE error."""
    diffs = np.abs(truth - predictions)
    return ((diffs) ** 2).mean()


def _calc_rmse(
    self,
    predictions: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Calculates RMSE error."""
    diffs = np.abs(truth - predictions)
    return np.sqrt(_calc_mse(self, predictions, truth))


def _calc_mase(
    self,
    predictions: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Calculates MASE error.
    mean(|actual - forecast| / naiveError), where
    naiveError = 1/ (n-1) sigma^n_[i=2](|actual_[i] - actual_[i-1]|)
    """
    diffs = np.abs(truth - predictions)
    naive_error = np.abs(np.diff(truth)).sum() / (truth.shape[0] - 1)
    return diffs.mean() / naive_error


def _calc_mape(
    self,
    predictions: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Calculates MAPE error."""
    return np.mean(np.abs((truth - predictions) / truth))


def _calc_smape(
    self,
    predictions: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Calculates SMAPE error."""
    return ((abs(truth - predictions) / (truth + predictions)).sum()) * (2.0 / truth.size)


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
        model = self.model_class(self.params)
        df_train, df_val = df_utils.split_df(
            df=self.data.df,
            n_lags=0,
            n_forecasts=1,
            valid_p=self.test_percentage / 100.0,
        )
        df_val = model.maybe_add_first_inputs_to_df(df_train, df_val)
        model.fit(df=df_train, freq=self.data.freq)
        fcst_train = model.predict(df_train)
        fcst_val = model.predict(df_val)
        result_train = self.experiment_name.copy()
        result_val = self.experiment_name.copy()

        for metric in self.metrics:
            metric_train = self.error_funcs[metric](self, df_train["y"], fcst_train["fcst"])
            metric_val = self.error_funcs[metric](self, df_val["y"], fcst_val["fcst"])
            result_train[metric] = metric_train
            result_val[metric] = metric_val
        return result_train, result_val


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

        metrics_train = pd.DataFrame(columns=self.metrics)
        metrics_val = pd.DataFrame(columns=self.metrics)
        for df_train, df_val in folds:
            model = self.model_class(self.params)
            df_val = model.maybe_add_first_inputs_to_df(df_train, df_val)
            model.fit(df=df_train, freq=self.data.freq)
            fcst_train = model.predict(df_train)
            fcst_val = model.predict(df_val)

            errors_train = []
            errors_val = []
            for metric in self.metrics:
                errors_train.append(self.error_funcs[metric](self, df_train["y"], fcst_train["fcst"]))
                errors_val.append(self.error_funcs[metric](self, df_val["y"], fcst_val["fcst"]))
            metrics_train.loc[len(metrics_train.index)] = errors_train
            metrics_val.loc[len(metrics_train.index)] = errors_val

        result_train = self.experiment_name.copy()
        result_val = self.experiment_name.copy()
        for metric in self.metrics:
            result_train[metric] = metrics_train[metric].tolist()
            result_val[metric] = metrics_val[metric].tolist()
        return result_train, result_val


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
        self.experiments = []
        for ts in self.datasets:
            for model_class, params in self.model_classes_and_params:
                exp = SimpleExperiment(
                    model_class=model_class,
                    params=params,
                    data=ts,
                    metrics=self.metrics,
                    test_percentage=self.test_percentage,
                )
                self.experiments.append(exp)

    def run(self):
        self.setup_experiments()
        cols = list(self.experiments[0].experiment_name.keys()) + self.metrics
        results_train = pd.DataFrame(columns=cols)
        results_val = pd.DataFrame(columns=cols)
        for exp in self.experiments:
            exp.metrics = self.metrics
            res_train, res_val = exp.run()
            results_train = results_train.append(res_train, ignore_index=True)
            results_val = results_val.append(res_val, ignore_index=True)
        return results_train, results_val


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
        self.experiments = []
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
                self.experiments.append(exp)

    def run(self):
        results_train, results_val = super().run()
        val = results_val.copy(deep=True)
        train = results_train.copy(deep=True)
        results_summary = results_val.copy(deep=True).drop(self.metrics, axis=1)
        for metric in self.metrics:
            results_summary["train_" + metric] = train[metric].apply(lambda x: np.array(x).mean())
            results_summary["train_" + metric + "_std"] = train[metric].apply(lambda x: np.array(x).std())
        for metric in self.metrics:
            results_summary["val_" + metric] = val[metric].apply(lambda x: np.array(x).mean())
            results_summary["val_" + metric + "_std"] = val[metric].apply(lambda x: np.array(x).std())
        return results_summary, results_train, results_val


def debug_experiment():
    import os
    import pathlib

    DIR = pathlib.Path(__file__).parent.parent.absolute()
    DATA_DIR = os.path.join(DIR, "tests", "test-data")
    AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
    air_passengers_df = pd.read_csv(AIR_FILE)

    ts = Dataset(df=air_passengers_df, name="air_passengers", freq="MS")
    params = {"seasonality_mode": "multiplicative"}
    exp = SimpleExperiment(
        model_class=ProphetModel,
        params=params,
        data=ts,
        metrics=["MAE", "MSE", "MASE", "RMSE"],
        test_percentage=25,
    )
    result_train, result_val = exp.run()
    print(result_val)

    ts = Dataset(df=air_passengers_df, name="air_passengers", freq="MS")
    params = {"seasonality_mode": "multiplicative", "train_speed": 2}
    exp_cv = CrossValidationExperiment(
        model_class=NeuralProphetModel,
        params=params,
        data=ts,
        metrics=["MAE", "MSE", "MASE", "RMSE", "MAPE", "SMAPE"],
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
        (NeuralProphetModel, {"train_speed": 2}),
        # (NeuralProphetModel, {"n_changepoints": 5}),
        # (NeuralProphetModel, {"seasonality_mode": "multiplicative", "learning_rate": 0.1}),
    ]
    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,  # iterate over this list of tuples
        datasets=dataset_list,  # iterate over this list
        metrics=["MAE", "MSE"],
        test_percentage=25,
    )
    results_train, results_val = benchmark.run()
    print(results_val)

    benchmark_cv = CrossValidationBenchmark(
        model_classes_and_params=model_classes_and_params,  # iterate over this list of tuples
        datasets=dataset_list,  # iterate over this list
        metrics=["MAE", "MSE"],
        test_percentage=10,
        num_folds=3,
        fold_overlap_pct=0,
    )
    results_summary, results_train, results_val = benchmark_cv.run()
    print(results_summary)


if __name__ == "__main__":
    debug_experiment()
    # debug_benchmark()
