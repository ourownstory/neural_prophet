import os
import gc
from dataclasses import dataclass, field
from typing import List, Generic, Optional, TypeVar, Tuple, Type
from abc import ABC, abstractmethod
import logging

import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet, df_utils
from multiprocessing.pool import Pool


try:
    from prophet import Prophet

    _prophet_installed = True
except ImportError:
    Prophet = None
    _prophet_installed = False


log = logging.getLogger("NP.benchmark")
log.warning(
    "Benchmarking Framework is not covered by tests. Please report any bugs you find."
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
    progress_bar: bool = False

    def __post_init__(self):
        self.model = self.model_class(**self.params)
        self.n_forecasts = self.model.n_forecasts
        self.n_lags = self.model.n_lags

    def fit(self, df: pd.DataFrame, freq: str):
        self.freq = freq
        _ = self.model.fit(df=df, freq=freq, progress_bar=self.progress_bar, minimal=True)

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
        fcst_train, df_train = model.maybe_drop_first_forecasts(fcst_train, df_train)
        fcst_test, df_test = model.maybe_drop_first_forecasts(fcst_test, df_test)

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
                pool.map_async(self._run_fold, args, callback=self._log_results)
                pool.close()
                pool.join()
            gc.collect()
        else:
            for current_fold, (df_train, df_test) in enumerate(folds):
                args = (df_train, df_test, current_fold)
                self._log_results(self._run_fold(args))

        results_cv_test_df = pd.DataFrame()
        results_cv_train_df = pd.DataFrame()
        results_cv_test_df = results_cv_test_df.append(self.results_cv_test, ignore_index=True)
        results_cv_train_df = results_cv_train_df.append(self.results_cv_test, ignore_index=True)
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

    @abstractmethod
    def setup_experiments(self):
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
                pool.map_async(self._run_exp, args_list, callback=self._log_result)
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

    def write_summary_to_csv(self, df_summary):
        model_name = self.model_classes_and_params[0][0].model_name
        params = "".join(["_{0}_{1}".format(k, v) for k, v in self.model_classes_and_params[0][1].items()])
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        name = "metric_summary_" + model_name + params + ".csv"
        print(name)
        df_summary.to_csv(os.path.join(self.save_dir, name), encoding="utf-8", index=False)

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
        self.write_summary_to_csv(df_metrics_summary)
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
    num_processes: int = 1

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
    num_processes: int = 1

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


def debug_simple_experiment():
    log.info("debug_experiment")
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
    log.info("SimpleExperiment")
    exp = SimpleExperiment(
        model_class=NeuralProphetModel,
        params=params,
        data=ts,
        metrics=list(ERROR_FUNCTIONS.keys()),
        test_percentage=25,
        save_dir=SAVE_DIR,
    )
    result_train, result_val = exp.run()
    print(result_val)
    print("#### Done with debug_simple_experiment")


def debug_cv_experiment(pool=None):
    log.info("debug_experiment")
    import os
    import pathlib

    DIR = pathlib.Path(__file__).parent.parent.absolute()
    DATA_DIR = os.path.join(DIR, "tests", "test-data")
    AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
    air_passengers_df = pd.read_csv(AIR_FILE)
    SAVE_DIR = "test_benchmark_logging"

    ts = Dataset(df=air_passengers_df, name="air_passengers", freq="MS")
    params = {
        # "seasonality_mode": "multiplicative",
    }
    log.info("CrossValidationExperiment")
    exp_cv = CrossValidationExperiment(
        model_class=NeuralProphetModel,
        params=params,
        data=ts,
        metrics=list(ERROR_FUNCTIONS.keys()),
        test_percentage=10,
        num_folds=2,
        fold_overlap_pct=0,
        save_dir=SAVE_DIR,
        # num_processes=1,
    )
    result_train, result_val = exp_cv.run()
    print(result_val)
    print("#### Done with debug_cv_experiment")


def debug_manual_benchmark():
    log.info("debug_manual_benchmark")
    import os
    import pathlib

    DIR = pathlib.Path(__file__).parent.parent.absolute()
    DATA_DIR = os.path.join(DIR, "tests", "test-data")
    PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
    AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
    air_passengers_df = pd.read_csv(AIR_FILE)
    peyton_manning_df = pd.read_csv(PEYTON_FILE)[:1000]
    SAVE_DIR = "test_benchmark_logging"

    metrics = list(ERROR_FUNCTIONS.keys())

    log.info("ManualBenchmark")
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
    benchmark = ManualBenchmark(experiments=experiments, metrics=metrics, num_processes=4)
    results_train, results_test = benchmark.run()
    print(results_test.to_string())
    print("#### Done with debug_manual_benchmark")


def debug_manual_cv_benchmark():
    log.info("debug_manual_benchmark")
    import os
    import pathlib

    DIR = pathlib.Path(__file__).parent.parent.absolute()
    DATA_DIR = os.path.join(DIR, "tests", "test-data")
    PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")
    AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
    air_passengers_df = pd.read_csv(AIR_FILE)
    peyton_manning_df = pd.read_csv(PEYTON_FILE)[:1000]
    SAVE_DIR = "test_benchmark_logging"

    metrics = list(ERROR_FUNCTIONS.keys())

    log.info("ManualCVBenchmark")
    experiments = [
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={"seasonality_mode": "multiplicative", "learning_rate": 0.1},
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=10,
            num_folds=2,
            fold_overlap_pct=0,
            save_dir=SAVE_DIR,
        ),
        CrossValidationExperiment(
            model_class=ProphetModel,
            params={
                "seasonality_mode": "multiplicative",
            },
            data=Dataset(df=peyton_manning_df, name="peyton_manning", freq="D"),
            metrics=metrics,
            test_percentage=10,
            num_folds=3,
            fold_overlap_pct=0,
            save_dir=SAVE_DIR,
            num_processes=1,
        ),
        CrossValidationExperiment(
            model_class=NeuralProphetModel,
            params={
                "seasonality_mode": "multiplicative",
            },
            data=Dataset(df=air_passengers_df, name="air_passengers", freq="MS"),
            metrics=metrics,
            test_percentage=10,
            num_folds=1,
            fold_overlap_pct=0,
            save_dir=SAVE_DIR,
        ),
    ]
    benchmark_cv = ManualCVBenchmark(experiments=experiments, metrics=metrics, num_processes=3)
    results_summary, results_train, results_test = benchmark_cv.run()
    print(results_summary.to_string())
    print(results_train.to_string())
    print(results_test.to_string())
    print("#### Done with debug_manual_cv_benchmark")


def debug_simple_benchmark():
    log.info("debug_simple_benchmark")
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
        # (ProphetModel, {"seasonality_mode": "multiplicative"}),
        # (NeuralProphetModel, {"learning_rate": 0.1}),
        (ProphetModel, {}),
        # (NeuralProphetModel, {"seasonality_mode": "multiplicative", "learning_rate": 0.1}),
    ]
    log.info("SimpleBenchmark")
    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,  # iterate over this list of tuples
        datasets=dataset_list,  # iterate over this list
        metrics=list(ERROR_FUNCTIONS.keys()),
        test_percentage=25,
        save_dir=SAVE_DIR,
        num_processes=3,
    )
    results_train, results_test = benchmark.run()
    print(results_test.to_string())
    print("#### Done with debug_simple_benchmark")


def debug_cv_benchmark():
    log.info("debug_simple_benchmark")
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
        Dataset(df=peyton_manning_df[:1000], name="peyton_manning", freq="D"),
        # Dataset(df = retail_sales_df, name = "retail_sales", freq = "D"),
        # Dataset(df = yosemite_temps_df, name = "yosemite_temps", freq = "5min"),
        # Dataset(df = ercot_load_df, name = "ercot_load", freq = "H"),
    ]
    model_classes_and_params = [
        # (NeuralProphetModel, {"seasonality_mode": "multiplicative", "learning_rate": 0.1}),
        # (ProphetModel, {"seasonality_mode": "multiplicative"}),
        (NeuralProphetModel, {"learning_rate": 0.1, "seasonality_mode": "multiplicative"}),
        # (ProphetModel, {}),
        # (NeuralProphetModel, {"seasonality_mode": "multiplicative", "learning_rate": 0.1}),
    ]

    log.info("CrossValidationBenchmark multi")
    benchmark_cv = CrossValidationBenchmark(
        model_classes_and_params=model_classes_and_params,  # iterate over this list of tuples
        datasets=dataset_list,  # iterate over this list
        metrics=list(ERROR_FUNCTIONS.keys()),
        test_percentage=10,
        num_folds=5,
        fold_overlap_pct=0,
        save_dir=SAVE_DIR,
        num_processes=5,
    )
    results_summary, results_train, results_test = benchmark_cv.run()
    print(results_summary.to_string())
    print(results_train.to_string())
    print(results_test.to_string())
    print("#### Done with debug_cv_benchmark")


if __name__ == "__main__":
    # debug_simple_experiment()
    # debug_cv_experiment()
    # debug_manual_benchmark()
    # debug_manual_cv_benchmark()
    # debug_simple_benchmark()
    debug_cv_benchmark()
