import datetime
import gc
import logging
import math
import os
from copy import copy, deepcopy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multiprocessing.pool import Pool
from typing import List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet, df_utils

try:
    from prophet import Prophet

    _prophet_installed = True
except ImportError:
    Prophet = None
    _prophet_installed = False

log = logging.getLogger("NP.benchmark")
log.debug(
    "Note: The benchmarking framework is not properly documented."
    "Please help us by reporting any bugs and adding documentation."
    "Multiprocessing is not covered by tests and may break on your device."
    "If you use multiprocessing, only run one benchmark per python script."
)


def _calc_mae(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray = None,
) -> float:
    """Calculates MAE error."""
    error_abs = np.abs(np.subtract(truth, predictions))
    return 1.0 * np.nanmean(error_abs, dtype="float32")


def _calc_mse(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray = None,
) -> float:
    """Calculates MSE error."""
    error_squared = np.square(np.subtract(truth, predictions))
    return 1.0 * np.nanmean(error_squared, dtype="float32")


def _calc_rmse(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray = None,
) -> float:
    """Calculates RMSE error."""
    mse = _calc_mse(predictions, truth)
    return np.sqrt(mse)


def _calc_mase(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray,
) -> float:
    """Calculates MASE error.
    according to https://robjhyndman.com/papers/mase.pdf
    Note: Naive error is computed over in-sample data.
        MASE = MAE / NaiveMAE,
    where: MAE = mean(|actual - forecast|)
    where: NaiveMAE = mean(|actual_[i] - actual_[i-1]|)
    """
    assert len(truth_train) > 1
    mae = _calc_mae(predictions, truth)
    naive_mae = _calc_mae(np.array(truth_train[:-1]), np.array(truth_train[1:]))
    return np.divide(mae, 1e-9 + naive_mae)


def _calc_rmsse(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray,
) -> float:
    """Calculates RMSSE error.
    according to https://robjhyndman.com/papers/mase.pdf
    Note: Naive error is computed over in-sample data.
    MSSE = RMSE / NaiveRMSE,
    where: RMSE = sqrt(mean((actual - forecast)^2))
    where: NaiveMSE = sqrt(mean((actual_[i] - actual_[i-1])^2))
    """
    assert len(truth_train) > 1
    rmse = _calc_rmse(predictions, truth)
    naive_rmse = _calc_rmse(np.array(truth_train[:-1]), np.array(truth_train[1:]))
    return np.divide(rmse, 1e-9 + naive_rmse)


def _calc_mape(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray = None,
) -> float:
    """Calculates MAPE error."""
    error = np.subtract(truth, predictions)
    error_relative = np.abs(np.divide(error, truth))
    return 100.0 * np.nanmean(error_relative, dtype="float32")


def _calc_smape(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray = None,
) -> float:
    """Calculates SMAPE error."""
    absolute_error = np.abs(np.subtract(truth, predictions))
    absolute_sum = np.abs(truth) + np.abs(predictions)
    error_relative_sym = np.divide(absolute_error, absolute_sum)
    return 100.0 * np.nanmean(error_relative_sym, dtype="float32")


ERROR_FUNCTIONS = {
    "MAE": _calc_mae,
    "MSE": _calc_mse,
    "RMSE": _calc_rmse,
    "MASE": _calc_mase,
    "RMSSE": _calc_rmsse,
    "MAPE": _calc_mape,
    "SMAPE": _calc_smape,
}


def convert_to_datetime(series):
    if series.isnull().any():
        raise ValueError("Found NaN in column ds.")
    if series.dtype == np.int64:
        series = series.astype(str)
    if not np.issubdtype(series.dtype, np.datetime64):
        series = pd.to_datetime(series)
    if series.dt.tz is not None:
        raise ValueError("Column ds has timezone specified, which is not supported. Remove timezone.")
    return series


@dataclass
class Dataset:
    """
    example use:
    >>> dataset = Dataset(
    >>>     df = pd.read_csv('air_passengers.csv'),
    >>>     name = "air_passengers",
    >>>     freq = "MS",
    >>>     seasonalities = [1, 7] # daily and weekly seasonality
    >>> ),
    """

    df: pd.DataFrame
    name: str
    freq: str
    seasonalities: List = field(default_factory=list)


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

    def maybe_drop_added_dates(self, predicted, df):
        """if Model imputed any dates: removes any dates in predicted which are not in df_test."""
        return predicted.reset_index(drop=True), df.reset_index(drop=True)


@dataclass
class ProphetModel(Model):
    model_name: str = "Prophet"
    model_class: Type = Prophet

    def __post_init__(self):
        if not _prophet_installed:
            raise RuntimeError("Requires prophet to be installed")
        data_params = self.params["_data_params"]
        if len(data_params) != 0:
            if "seasonalities" in data_params:
                seasonalities = data_params["seasonalities"]
                custom_seasonalities = []
                if len(seasonalities) > 0:
                    self.params.update({"daily_seasonality": False})
                    self.params.update({"weekly_seasonality": False})
                    self.params.update({"yearly_seasonality": False})
                for season_days in seasonalities:
                    if math.isclose(season_days, 1):
                        self.params.update({"daily_seasonality": True})
                    elif math.isclose(season_days, 7):
                        self.params.update({"weekly_seasonality": True})
                    elif math.isclose(season_days, 365) or math.isclose(season_days, 365.25):
                        self.params.update({"yearly_seasonality": True})
                    else:
                        custom_seasonalities.append(season_days)
        model_params = deepcopy(self.params)
        model_params.pop("_data_params")
        self.model = self.model_class(**model_params)
        for seasonality in custom_seasonalities:
            self.model.add_seasonality(name="{}_daily".format(str(seasonality)), period=seasonality)
        self.n_forecasts = 1
        self.n_lags = 0

    def fit(self, df: pd.DataFrame, freq: str):
        self.freq = freq
        self.model = self.model.fit(df=df)

    def predict(self, df: pd.DataFrame):
        fcst = self.model.predict(df=df)
        fcst_df = pd.DataFrame({"time": fcst.ds, "y": df.y, "yhat1": fcst.yhat})
        return fcst_df


@dataclass
class NeuralProphetModel(Model):
    model_name: str = "NeuralProphet"
    model_class: Type = NeuralProphet

    def __post_init__(self):
        data_params = self.params["_data_params"]
        custom_seasonalities = []
        if len(data_params) != 0:
            if "seasonalities" in data_params:
                seasonalities = data_params["seasonalities"]
                if len(seasonalities) > 0:
                    self.params.update({"daily_seasonality": False})
                    self.params.update({"weekly_seasonality": False})
                    self.params.update({"yearly_seasonality": False})
                for season_days in seasonalities:
                    if math.isclose(season_days, 1):
                        self.params.update({"daily_seasonality": True})
                    elif math.isclose(season_days, 7):
                        self.params.update({"weekly_seasonality": True})
                    elif math.isclose(season_days, 365) or math.isclose(season_days, 365.25):
                        self.params.update({"yearly_seasonality": True})
                    else:
                        custom_seasonalities.append(season_days)
        model_params = deepcopy(self.params)
        model_params.pop("_data_params")
        self.model = self.model_class(**model_params)
        for seasonality in custom_seasonalities:
            self.model.add_seasonality(name=str(seasonality), period=6)
        self.n_forecasts = self.model.n_forecasts
        self.n_lags = self.model.n_lags

    def fit(self, df: pd.DataFrame, freq: str):
        self.freq = freq
        _ = self.model.fit(df=df, freq=freq, progress="none", minimal=True)

    def predict(self, df: pd.DataFrame):
        fcst = self.model.predict(df=df)
        y_cols = ["y"] + [col for col in fcst.columns if "yhat" in col]
        fcst_df = pd.DataFrame({"time": fcst.ds})
        for y_col in y_cols:
            fcst_df[y_col] = fcst[y_col]
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

    def maybe_drop_added_dates(self, predicted, df):
        """if Model imputed any dates: removes any dates in predicted which are not in df_test."""
        df["ds"] = convert_to_datetime(df["ds"])
        df.set_index("ds")
        predicted.set_index("time")
        predicted = predicted.loc[df.index]
        predicted = predicted.reset_index()
        df = df.reset_index()
        return predicted, df


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
    num_processes: int = 1

    def __post_init__(self):
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {
                "data": self.data.name,
                "model": self.model_class.model_name,
                "params": str(self.params),
            }
        data_params = {}
        if len(self.data.seasonalities) > 0:
            data_params["seasonalities"] = self.data.seasonalities
        self.params.update({"_data_params": data_params})
        if not hasattr(self, "experiment_name") or self.experiment_name is None:
            self.experiment_name = "{}_{}{}".format(
                self.data.name,
                self.model_class.model_name,
                "".join(["_{0}_{1}".format(k, v) for k, v in self.params.items()]),
            )

    def write_results_to_csv(self, df, prefix, current_fold=None):
        # save fcst and create dir if necessary
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        name = self.experiment_name
        if current_fold is not None:
            name = name + "_fold_" + str(current_fold)
        name = prefix + "_" + name + ".csv"
        df.to_csv(os.path.join(self.save_dir, name), encoding="utf-8", index=False)

    def _evaluate_model(self, model, df_train, df_test, current_fold=None):
        df_test = model.maybe_add_first_inputs_to_df(df_train, df_test)
        min_length = model.n_lags + model.n_forecasts
        if min_length > len(df_train):
            raise ValueError("Not enough training data to create a single input sample.")
        elif len(df_train) - min_length < 5:
            log.warning("Less than 5 training samples")
        if min_length > len(df_test):
            raise ValueError("Not enough test data to create a single input sample.")
        elif len(df_test) - min_length < 5:
            log.warning("Less than 5 test samples")
        fcst_train = model.predict(df_train)
        fcst_test = model.predict(df_test)
        # remove added input lags
        fcst_train, df_train = model.maybe_drop_first_forecasts(fcst_train, df_train)
        fcst_test, df_test = model.maybe_drop_first_forecasts(fcst_test, df_test)
        # remove interpolated dates
        fcst_train, df_train = model.maybe_drop_added_dates(fcst_train, df_train)
        fcst_test, df_test = model.maybe_drop_added_dates(fcst_test, df_test)

        result_train = self.metadata.copy()
        result_test = self.metadata.copy()
        for metric in self.metrics:
            # todo: parallelize
            n_yhats_train = sum(["yhat" in colname for colname in fcst_train.columns])
            n_yhats_test = sum(["yhat" in colname for colname in fcst_test.columns])

            assert n_yhats_train == n_yhats_test, "Dimensions of fcst dataframe faulty."

            metric_train_list = []
            metric_test_list = []

            fcst_train = fcst_train.fillna(value=np.nan)
            df_train = df_train.fillna(value=np.nan)
            fcst_test = fcst_test.fillna(value=np.nan)
            df_test = df_test.fillna(value=np.nan)

            for x in range(1, n_yhats_train + 1):
                metric_train_list.append(
                    ERROR_FUNCTIONS[metric](
                        predictions=fcst_train["yhat{}".format(x)].values,
                        truth=df_train["y"].values,
                        truth_train=df_train["y"].values,
                    )
                )
                metric_test_list.append(
                    ERROR_FUNCTIONS[metric](
                        predictions=fcst_test["yhat{}".format(x)].values,
                        truth=df_test["y"].values,
                        truth_train=df_train["y"].values,
                    )
                )
            result_train[metric] = np.nanmean(metric_train_list, dtype="float32")
            result_test[metric] = np.nanmean(metric_test_list, dtype="float32")

        if self.save_dir is not None:
            self.write_results_to_csv(fcst_train, prefix="predicted_train", current_fold=current_fold)
            self.write_results_to_csv(fcst_test, prefix="predicted_test", current_fold=current_fold)
        del fcst_train
        del fcst_test
        gc.collect()
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

    # results_cv_train: dict = field(init=False)
    # results_cv_test: dict = field(init=False)

    def _run_fold(self, args):
        df_train, df_test, current_fold = args
        model = self.model_class(self.params)
        model.fit(df=df_train, freq=self.data.freq)
        result_train, result_test = self._evaluate_model(model, df_train, df_test, current_fold=current_fold)
        del model
        gc.collect()
        return (result_train, result_test)

    def _log_results(self, results):
        if type(results) != list:
            results = [results]
        for res in results:
            result_train, result_test = res
            for m in self.metrics:
                self.results_cv_train[m].append(result_train[m])
                self.results_cv_test[m].append(result_test[m])

    def _log_error(self, error):
        log.error(repr(error))

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
        self.results_cv_train = self.metadata.copy()
        self.results_cv_test = self.metadata.copy()
        for m in self.metrics:
            self.results_cv_train[m] = []
            self.results_cv_test[m] = []
        if self.num_processes > 1 and self.num_folds > 1:
            with Pool(self.num_processes) as pool:
                args = [(df_train, df_test, current_fold) for current_fold, (df_train, df_test) in enumerate(folds)]
                pool.map_async(self._run_fold, args, callback=self._log_results, error_callback=self._log_error)
                pool.close()
                pool.join()
            gc.collect()
        else:
            for current_fold, (df_train, df_test) in enumerate(folds):
                args = (df_train, df_test, current_fold)
                self._log_results(self._run_fold(args))

        if self.save_dir is not None:
            results_cv_test_df = pd.DataFrame()
            results_cv_train_df = pd.DataFrame()
            results_cv_test_df = results_cv_test_df.append(self.results_cv_test, ignore_index=True)
            results_cv_train_df = results_cv_train_df.append(self.results_cv_train, ignore_index=True)
            self.write_results_to_csv(results_cv_test_df, prefix="summary_test")
            self.write_results_to_csv(results_cv_train_df, prefix="summary_train")

        return self.results_cv_train, self.results_cv_test


@dataclass
class Benchmark(ABC):
    """Abstract Benchmarking class"""

    metrics: List[str]

    # df_metrics_train: pd.DataFrame = field(init=False)
    # df_metrics_test: pd.DataFrame = field(init=False)

    def __post_init__(self):
        if not hasattr(self, "experiments"):
            self.experiments = self.setup_experiments()
        if not hasattr(self, "num_processes"):
            self.num_processes = 1
        if not hasattr(self, "save_dir"):
            self.save_dir = None

    def setup_experiments(self):
        if self.save_dir is not None:
            for e in self.experiments:
                if e.save_dir is None:
                    e.save_dir = self.save_dir
        return self.experiments

    # def _run_exp(self, exp, verbose=False, exp_num=0):
    def _run_exp(self, args):
        exp, verbose, exp_num = args
        if verbose:
            log.info("--------------------------------------------------------")
            log.info("starting exp {}: {}".format(exp_num, exp.experiment_name))
            log.info("--------------------------------------------------------")
        exp.metrics = self.metrics
        res_train, res_test = exp.run()
        if verbose:
            log.info("--------------------------------------------------------")
            log.info("finished exp {}: {}".format(exp_num, exp.experiment_name))
            log.info("test results {}: {}".format(exp_num, res_test))
            log.info("--------------------------------------------------------")
        # del exp
        # gc.collect()
        return (res_train, res_test)

    def _log_result(self, results):
        if type(results) != list:
            results = [results]
        for res in results:
            res_train, res_test = res
            self.df_metrics_train = self.df_metrics_train.append(res_train, ignore_index=True)
            self.df_metrics_test = self.df_metrics_test.append(res_test, ignore_index=True)

    def _log_error(self, error):
        log.error(repr(error))

    def run(self, verbose=True):
        # setup DataFrame to store each experiment in a row
        cols = list(self.experiments[0].metadata.keys()) + self.metrics
        self.df_metrics_train = pd.DataFrame(columns=cols)
        self.df_metrics_test = pd.DataFrame(columns=cols)

        if verbose:
            log.info("Experiment list:")
            for i, exp in enumerate(self.experiments):
                log.info("exp {}/{}: {}".format(i + 1, len(self.experiments), exp.experiment_name))
        log.info("---- Staring Series of {} Experiments ----".format(len(self.experiments)))
        if self.num_processes > 1 and len(self.experiments) > 1:
            if not all([exp.num_processes == 1 for exp in self.experiments]):
                raise ValueError("can not set multiprocessing in experiments and Benchmark.")
            with Pool(self.num_processes) as pool:
                args_list = [(exp, verbose, i + 1) for i, exp in enumerate(self.experiments)]
                pool.map_async(self._run_exp, args_list, callback=self._log_result, error_callback=self._log_error)
                pool.close()
                pool.join()
            gc.collect()
        else:
            args_list = [(exp, verbose, i + 1) for i, exp in enumerate(self.experiments)]
            for args in args_list:
                self._log_result(self._run_exp(args))
                gc.collect()

        return self.df_metrics_train, self.df_metrics_test


@dataclass
class CVBenchmark(Benchmark, ABC):
    """Abstract Crossvalidation Benchmarking class"""

    def write_summary_to_csv(self, df_summary, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        models = [
            "{}-{}".format(e.metadata["model"], "".join(["_{0}_{1}".format(k, v) for k, v in e.params.items()]))
            for e in self.experiments
        ]
        models = "_".join(list(set(models)))
        stamp = str(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S-%f"))
        name = "metrics_summary_" + models + stamp + ".csv"
        log.debug(name)
        df_summary.to_csv(os.path.join(save_dir, name), encoding="utf-8", index=False)

    def _summarize_cv_metrics(self, df_metrics, name=None):
        df_metrics_summary = df_metrics.copy(deep=True)
        name = "" if name is None else "_{}".format(name)
        for metric in self.metrics:
            df_metrics_summary[metric + name] = df_metrics[metric].copy(deep=True).apply(lambda x: np.array(x).mean())
            df_metrics_summary[metric + "_std" + name] = (
                df_metrics[metric].copy(deep=True).apply(lambda x: np.array(x).std())
            )
        return df_metrics_summary

    def run(self, verbose=True):
        df_metrics_train, df_metrics_test = super().run(verbose=verbose)
        df_metrics_summary_train = self._summarize_cv_metrics(df_metrics_train)
        df_metrics_summary_train["split"] = "train"
        df_metrics_summary_test = self._summarize_cv_metrics(df_metrics_test)
        df_metrics_summary_test["split"] = "test"
        df_metrics_summary = df_metrics_summary_train.append(df_metrics_summary_test)
        if self.save_dir is not None:
            self.write_summary_to_csv(df_metrics_summary, save_dir=self.save_dir)
        return df_metrics_summary, df_metrics_train, df_metrics_test


@dataclass
class ManualBenchmark(Benchmark):
    """Manual Benchmarking class
    use example:
    >>> benchmark = ManualBenchmark(
    >>>     metrics=["MAE", "MSE"],
    >>>     experiments=experiment_list, # iterate over this list of experiments
    >>>     save_dir="./logs"
    >>> )
    >>> results_train, results_val = benchmark.run()
    """

    save_dir: Optional[str] = None
    experiments: List[Experiment] = None
    num_processes: int = 1


@dataclass
class ManualCVBenchmark(CVBenchmark):
    """Manual Crossvalidation Benchmarking class
    use example:
    >>> benchmark = ManualCVBenchmark(
    >>>     metrics=["MAE", "MSE"],
    >>>     experiments=cv_experiment_list, # iterate over this list of experiments
    >>>     save_dir="./logs"
    >>> )
    >>> results_train, results_val = benchmark.run()
    """

    save_dir: Optional[str] = None
    experiments: List[Experiment] = None
    num_processes: int = 1


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
    num_processes: int = 1

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
    num_processes: int = 1

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
                    num_processes=1,
                )
                experiments.append(exp)
        return experiments
