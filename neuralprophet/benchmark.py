from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet


NeuralProphetModel = NeuralProphet


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
    def initialize(self):
        pass


@dataclass
class NeuralProphetModel(Model):
    model_name: str = "NeuralProphet"
    model_class: Type = NeuralProphet

    def initialize(self):
        return NeuralProphet(**self.params)


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
    def fit(self):
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
    >>> result_train, result_val = exp.fit()
    """

    def fit(self):
        model_class = self.model_class(self.params)
        model = model_class.initialize()
        df_train, df_val = model.split_df(
            df=self.data.df,
            freq=self.data.freq,
            valid_p=self.test_percentage / 100.0,
        )
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
    >>> result_train, result_train, result_val = exp.fit()
    """

    num_folds: int
    fold_overlap_pct: float = 0

    def fit(self):
        model_class = self.model_class(self.params)
        folds = model_class.initialize().crossvalidation_split_df(
            df=self.data.df,
            freq=self.data.freq,
            k=self.num_folds,
            fold_pct=self.test_percentage / 100.0,
            fold_overlap_pct=self.fold_overlap_pct / 100.0,
        )
        metrics_train = pd.DataFrame(columns=self.metrics)
        metrics_val = pd.DataFrame(columns=self.metrics)
        for df_train, df_val in folds:
            m = model_class.initialize()
            train = m.fit(df=df_train, freq=self.data.freq)
            val = m.test(df=df_val)
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
            res_train, res_val = exp.fit()
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
