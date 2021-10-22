from dataclasses import dataclass, field
from typing import List, Generic, Optional, TypeVar, Tuple, Type
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet, df_utils
from fbprophet import Prophet


NeuralProphetModel = NeuralProphet
ProphetModel = Prophet

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
    def test(self, df: pd.DataFrame):
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
        df_val = pd.concat([df_train.tail(5), df_val], ignore_index=True)
        return df_val

    def fit(self, df: pd.DataFrame, freq: str):
        self.freq = freq
        metrics = self.model.fit(df=df, freq=freq)
        return metrics

    def predict(self, df: pd.DataFrame):
        fcst = self.model.predict(df=df)
        if self.model.n_forecasts > 1:
            raise NotImplementedError
        self.fcst_df = pd.DataFrame({"time": fcst.ds, "fcst": fcst.yhat1})
        return self.fcst_df

    def test(self, df: pd.DataFrame):
        if self.model.n_lags > 0:
            raise NotImplementedError("data overbleed when using prophet")
        metrics = self.model.test(df=df)
        return metrics


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

    def test(self, df: pd.DataFrame):
        fcst = self.predict(df=df)
        return fcst


def calculate_metrics(fcst):
    pass


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
        model = self.model_class(self.params)
        df_train, df_val = df_utils.single_split_df(
            df=self.data.df,
            n_lags = 0,
            n_forecasts = 1,
            valid_p=self.test_percentage / 100.0,
            inputs_overbleed = True,
        )

        df_val = model.maybe_add_first_inputs_to_df(df_train, df_val)        
        metrics_train = model.fit(df=df_train, freq=self.data.freq)       
        metrics_val = model.test(df=df_val)
        result_train = self.experiment_name.copy()
        result_val = self.experiment_name.copy()
        for metric in self.metrics:
            result_train[metric] = metrics_train[metric].values[-1]
            result_val[metric] = metrics_val[metric].values[-1]
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
        folds = self.model_class(self.params).model.crossvalidation_split_df(
            df=self.data.df,
            freq=self.data.freq,
            k=self.num_folds,
            fold_pct=self.test_percentage / 100.0,
            fold_overlap_pct=self.fold_overlap_pct / 100.0,
        )
        metrics_train = pd.DataFrame(columns=self.metrics)
        metrics_val = pd.DataFrame(columns=self.metrics)
        for df_train, df_val in folds:
            model = self.model_class(self.params)
            train = model.fit(df=df_train, freq=self.data.freq)
            val = model.test(df=df_val)
            metrics_train = metrics_train.append(train[self.metrics].iloc[-1])
            metrics_val = metrics_val.append(val[self.metrics].iloc[-1])
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

    model_classes_and_params: List[tuple[Model, dict]]
    datasets: List[Dataset]
    metrics: list[str]
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
        model_class=NeuralProphetModel,
        params=params,
        data=ts,
        metrics=["MAE", "MSE"],
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
        metrics=["MAE", "MSE"],
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
    #debug_benchmark()
