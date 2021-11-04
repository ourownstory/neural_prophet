import os
from dataclasses import dataclass, field
from typing import List, Generic, Optional, TypeVar, Tuple, Type
from abc import ABC, abstractmethod
import logging

import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet, df_utils
import multiprocessing as mp


try:
    from prophet import Prophet

    _prophet_installed = True
except ImportError:
    Prophet = None
    _prophet_installed = False


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


ERROR_FUNCTIONS = {
    "MAE": _calc_mae,
    "MSE": _calc_mse,
    "RMSE": _calc_rmse,
    "MASE": _calc_mase,
    "MSSE": _calc_msse,
    "MAPE": _calc_mape,
    "SMAPE": _calc_smape,
}


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
        if Model with lags: removes first n_lags values from predicted and df_test
        else (time-features only): returns unchanged df_test
        """
        return predicted.reset_index(drop=True), df.reset_index(drop=True)


@dataclass
class ProphetModel(Model):
    model_name: str = "Prophet"
    model_class: Type = Prophet

    def __post_init__(self):
        if not _prophet_installed:
            raise RuntimeError("Requires prophet to be installed")
        self.model = self.model_class(**self.params)

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
    experiment_name: Optional[str] = None
    metadata: Optional[dict] = None
    save_dir: Optional[str] = None

    def __post_init__(self):
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {
                "data": self.data.name,
                "model": self.model_class.model_name,
                "params": str(self.params),
            }
        if not hasattr(self, "experiment_name") or self.experiment_name is None:
            self.experiment_name = "{}_{}{}".format(
                self.data.name,
                self.model_class.model_name,
                "".join(["_{0}_{1}".format(k, v) for k, v in self.params.items()]),
            )
        # if not hasattr(self, "save_dir"):
        #     self.save_dir = None
        self.current_fold = None

    def write_results_to_csv(self, df, prefix):
        # save fcst and create dir if necessary
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        name = self.experiment_name
        if self.current_fold is not None:
            name = name + "_fold_" + str(self.current_fold)
        name = prefix + "_" + name + ".csv"
        df.to_csv(os.path.join(self.save_dir, name), encoding="utf-8", index=False)

    def _evaluate_model(self, model, df_train, df_test):
        df_test = model.maybe_add_first_inputs_to_df(df_train, df_test)
        fcst_train = model.predict(df_train)
        fcst_test = model.predict(df_test)
        fcst_train, df_train = model.maybe_drop_first_forecasts(fcst_train, df_train)
        fcst_test, df_test = model.maybe_drop_first_forecasts(fcst_test, df_test)

        result_train = self.metadata.copy()
        result_test = self.metadata.copy()
        for metric in self.metrics:
            # todo: parallelize
            result_train[metric] = ERROR_FUNCTIONS[metric](fcst_train["yhat"], df_train["y"])
            result_test[metric] = ERROR_FUNCTIONS[metric](fcst_test["yhat"], df_test["y"])
        if self.save_dir is not None:
            self.write_results_to_csv(fcst_train, prefix="predicted_train")
            self.write_results_to_csv(fcst_test, prefix="predicted_test")
        return result_train, result_test

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
    >>>     save_dir='./benchmark_logging',
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
    >>>     save_dir="./benchmark_logging/",
    >>> )
    >>> result_train, result_train, result_val = exp.run()
    """

    num_folds: int = 5
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
        results_cv_train = self.metadata.copy()
        results_cv_test = self.metadata.copy()
        for m in self.metrics:
            results_cv_train[m] = []
            results_cv_test[m] = []
        self.current_fold = 1
        for df_train, df_test in folds:
            # todo: parallelize
            model = self.model_class(self.params)
            model.fit(df=df_train, freq=self.data.freq)
            result_train, result_test = self._evaluate_model(model, df_train, df_test)
            for m in self.metrics:
                results_cv_train[m].append(result_train[m])
                results_cv_test[m].append(result_test[m])
            self.current_fold += 1
        return results_cv_train, results_cv_test


@dataclass
class Benchmark(ABC):
    """Abstract Benchmarking class"""

    metrics: List[str]
    df_metrics_train: pd.DataFrame = field(init=False)
    df_metrics_test: pd.DataFrame = field(init=False)

    def __post_init__(self):
        if not hasattr(self, "experiments"):
            self.experiments = self.setup_experiments()

    @abstractmethod
    def setup_experiments(self):
        return self.experiments

    def run_exp(self, exp):
        exp.metrics = self.metrics
        res_train, res_test = exp.run()
        return res_train, res_test

    def log_result(self, result):
        self.df_metrics_train = self.df_metrics_train.append(result[0], ignore_index=True)
        self.df_metrics_test = self.df_metrics_test.append(result[1], ignore_index=True)

    def run(self):
        # setup DataFrame to store each experiment in a row
        cols = list(self.experiments[0].metadata.keys()) + self.metrics
        self.df_metrics_train = pd.DataFrame(columns=cols)
        self.df_metrics_test = pd.DataFrame(columns=cols)

        num_processes = len(self.experiments)
        pool = mp.Pool(processes=num_processes)

        for exp in self.experiments:
            pool.apply_async(self.run_exp, args=(exp,), callback=self.log_result)
        pool.close()
        pool.join()

        return self.df_metrics_train, self.df_metrics_test


@dataclass
class CVBenchmark(Benchmark, ABC):
    """Abstract Crossvalidation Benchmarking class"""

    def _summarize_cv_metrics(self, df_metrics, name=None):
        df_metrics_summary = df_metrics.copy(deep=True)
        name = "" if name is None else "_{}".format(name)
        for metric in self.metrics:
            df_metrics_summary[metric + name] = df_metrics[metric].copy(deep=True).apply(lambda x: np.array(x).mean())
            df_metrics_summary[metric + "_std" + name] = (
                df_metrics[metric].copy(deep=True).apply(lambda x: np.array(x).std())
            )
        return df_metrics_summary

    def run(self):
        df_metrics_train, df_metrics_test = super().run()
        df_metrics_summary_train = self._summarize_cv_metrics(df_metrics_train)
        df_metrics_summary_train["split"] = "train"
        df_metrics_summary_test = self._summarize_cv_metrics(df_metrics_test)
        df_metrics_summary_test["split"] = "test"
        df_metrics_summary = df_metrics_summary_train.append(df_metrics_summary_test)
        return df_metrics_summary, df_metrics_train, df_metrics_test


@dataclass
class ManualBenchmark(Benchmark):
    """Manual Benchmarking class
    use example:
    >>> benchmark = ManualBenchmark(
    >>>     metrics=["MAE", "MSE"],
    >>>     experiments=experiment_list, # iterate over this list of experiments
    >>> )
    >>> results_train, results_val = benchmark.run()
    """

    experiments: List[Experiment] = None

    def setup_experiments(self):
        return self.experiments


@dataclass
class ManualCVBenchmark(CVBenchmark):
    """Manual Crossvalidation Benchmarking class
    use example:
    >>> benchmark = ManualCVBenchmark(
    >>>     metrics=["MAE", "MSE"],
    >>>     experiments=cv_experiment_list, # iterate over this list of experiments
    >>> )
    >>> results_train, results_val = benchmark.run()
    """

    experiments: List[Experiment] = None

    def setup_experiments(self):
        return self.experiments


@dataclass
class SimpleBenchmark(Benchmark):
    """
    use example:
    >>> benchmark = SimpleBenchmark(
    >>>     model_classes_and_params=model_classes_and_params, # iterate over this list of tuples
    >>>     datasets=dataset_list, # iterate over this list
    >>>     metrics=["MAE", "MSE"],
    >>>     test_percentage=25,
    >>>     save_dir='./benchmark_logging',
    >>> )
    >>> results_train, results_val = benchmark.run()
    """

    model_classes_and_params: List[Tuple[Model, dict]]
    datasets: List[Dataset]
    test_percentage: float
    save_dir: Optional[str] = None

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
                    save_dir=self.save_dir,
                )
                experiments.append(exp)
        return experiments


@dataclass
class CrossValidationBenchmark(CVBenchmark):
    """
    example use:
    >>> benchmark_cv = CrossValidationBenchmark(
    >>>     metrics=["MAE", "MSE"],
    >>>     model_classes_and_params=model_classes_and_params, # iterate over this list of tuples
    >>>     datasets=dataset_list, # iterate over this list
    >>>     test_percentage=10,
    >>>     num_folds=3,
    >>>     fold_overlap_pct=0,
    >>>     save_dir="./benchmark_logging/",
    >>> )
    >>> results_summary, results_train, results_val = benchmark_cv.run()
    """

    model_classes_and_params: List[Tuple[Model, dict]]
    datasets: List[Dataset]
    test_percentage: float
    num_folds: int = 5
    fold_overlap_pct: float = 0
    save_dir: Optional[str] = None

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
                    save_dir=self.save_dir,
                )
                experiments.append(exp)
        return experiments


def debug_experiment():
    import os
    import pathlib

    DIR = pathlib.Path(__file__).parent.parent.absolute()
    DATA_DIR = os.path.join(DIR, "tests", "test-data")
    AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
    air_passengers_df = pd.read_csv(AIR_FILE)
    SAVE_DIR = "test_benchmark_logging"

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
        save_dir=SAVE_DIR,
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
        save_dir=SAVE_DIR,
    )
    result_train, result_val = exp_cv.run()
    print(result_val)


def debug_manual_benchmark():
    import os
    import pathlib

    DIR = pathlib.Path(__file__).parent.parent.absolute()
    DATA_DIR = os.path.join(DIR, "tests", "test-data")
    PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
    AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
    air_passengers_df = pd.read_csv(AIR_FILE)
    peyton_manning_df = pd.read_csv(PEYTON_FILE)
    SAVE_DIR = "test_benchmark_logging"

    metrics = ["MAE", "MSE", "RMSE", "MASE", "MSSE", "MAPE", "SMAPE"]
    experiments = [
        SimpleExperiment(
            model_class=NeuralProphetModel,
            params={"seasonality_mode": "multiplicative", "learning_rate": 0.1},
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=25,
            save_dir=SAVE_DIR,
        ),
        SimpleExperiment(
            model_class=ProphetModel,
            params={
                "seasonality_mode": "multiplicative",
            },
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=25,
            save_dir=SAVE_DIR,
        ),
        SimpleExperiment(
            model_class=NeuralProphetModel,
            params={"learning_rate": 0.1},
            data=Dataset(df=peyton_manning_df, name="peyton_manning", freq="D"),
            metrics=metrics,
            test_percentage=15,
            save_dir=SAVE_DIR,
        ),
        SimpleExperiment(
            model_class=ProphetModel,
            params={},
            data=Dataset(df=peyton_manning_df, name="peyton_manning", freq="D"),
            metrics=metrics,
            test_percentage=15,
            save_dir=SAVE_DIR,
        ),
    ]
    benchmark = ManualBenchmark(experiments=experiments, metrics=metrics)
    results_train, results_test = benchmark.run()
    print(results_test.to_string())

    experiments = [
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={"seasonality_mode": "multiplicative", "learning_rate": 0.1},
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=10,
            num_folds=3,
            fold_overlap_pct=0,
            save_dir=SAVE_DIR,
        ),
        CrossValidationExperiment(
            model_class=ProphetModel,
            params={
                "seasonality_mode": "multiplicative",
            },
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=10,
            num_folds=3,
            fold_overlap_pct=0,
            save_dir=SAVE_DIR,
        ),
    ]
    benchmark_cv = ManualCVBenchmark(experiments=experiments, metrics=metrics)
    results_summary, results_train, results_test = benchmark_cv.run()
    print(results_summary.to_string())
    print(results_train.to_string())
    print(results_test.to_string())


def debug_simple_benchmark():
    import os
    import pathlib

    DIR = pathlib.Path(__file__).parent.parent.absolute()
    DATA_DIR = os.path.join(DIR, "tests", "test-data")
    PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
    AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
    YOS_FILE = os.path.join(DATA_DIR, "yosemite_temps.csv")
    SAVE_DIR = "test_benchmark_logging"

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
        save_dir=SAVE_DIR,
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
        save_dir=SAVE_DIR,
    )
    results_summary, results_train, results_test = benchmark_cv.run()
    print(results_summary.to_string())
    print(results_train.to_string())
    print(results_test.to_string())


if __name__ == "__main__":
    debug_experiment()
    debug_manual_benchmark()
    debug_simple_benchmark()
