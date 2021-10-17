import os
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


def load_data(path, file, columns=None):
    if columns is None:
        columns = ["ds", "y"]
    return pd.read_csv(os.path.join(path, file))[columns]


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
class Model:
    """
    example use:
    >>> models = []
    >>> for params in [{"n_changepoints": 5}, {"n_changepoints": 50},]:
    >>>     models.append(Model(
    >>>         model_name="NeuralProphet",
    >>>         model_class=NeuralProphet,
    >>>         params=params
    >>>     ))
    """

    model_name: str
    model_class: object
    params: dict

    def init(self):
        return self.model_class(**self.params)


@dataclass
class SimpleExperiment:
    """
    use example:
    >>> experiments = []
    >>> for d in datasets:
    >>>     for m in models:
    >>>         experiments.append(SimpleExperiment(model=m, dataset=d, valid_pct=0.2))
    """

    model: Model
    dataset: Dataset
    valid_pct: float
    experiment_name: dict = field(init=False)

    def __post_init__(self):
        self.experiment_name = {
            "data": self.dataset.name,
            "model": self.model.model_name,
            "params": str(self.model.params),
        }

    def fit(self, metrics):
        df_train, df_val = self.model.init().split_df(
            df=self.dataset.df,
            freq=self.dataset.freq,
            valid_p=self.valid_pct,
        )
        m = self.model.init()
        metrics_train = m.fit(df=df_train, freq=self.dataset.freq)
        metrics_val = m.test(df=df_val)
        result_train = self.experiment_name.copy()
        result_val = self.experiment_name.copy()
        for metric in metrics:
            result_train[metric] = metrics_train[metric].values[-1]
            result_val[metric] = metrics_val[metric].values[-1]
        return result_train, result_val


@dataclass
class SimpleBenchmark:
    """
    use example:
    >>> benchmark = SimpleBenchmark(
    >>>     experiments = experiments,
    >>>     metrics = ['MAE', 'MSE'])
    >>> results_train, results_val = benchmark.run()
    """

    experiments: list[SimpleExperiment]
    metrics: list[str]

    def run(self):
        cols = list(self.experiments[0].experiment_name.keys()) + self.metrics
        results_train = pd.DataFrame(columns=cols)
        results_val = pd.DataFrame(columns=cols)
        for exp in self.experiments:
            res_train, res_val = exp.fit(self.metrics)
            results_train = results_train.append(res_train, ignore_index=True)
            results_val = results_val.append(res_val, ignore_index=True)
        return results_train, results_val


@dataclass
class CVExperiment(SimpleExperiment):
    """
    >>> experiments_cv = []
    >>> for d in datasets:
    >>>     for m in models:
    >>>         experiments_cv.append(CVExperiment(
    >>>             model=m, dataset=d, n_folds=3, valid_pct=0.1))
    """

    n_folds: int
    fold_overlap_pct: float = 0

    def fit(self, metrics):
        folds = self.model.init().crossvalidation_split_df(
            df=self.dataset.df,
            freq=self.dataset.freq,
            k=self.n_folds,
            fold_pct=self.valid_pct,
            fold_overlap_pct=self.fold_overlap_pct,
        )
        metrics_train = pd.DataFrame(columns=metrics)
        metrics_val = pd.DataFrame(columns=metrics)
        for df_train, df_val in folds:
            m = self.model.init()
            train = m.fit(df=df_train, freq=self.dataset.freq)
            val = m.test(df=df_val)
            metrics_train = metrics_train.append(train[metrics].iloc[-1])
            metrics_val = metrics_val.append(val[metrics].iloc[-1])
        result_train = self.experiment_name.copy()
        result_val = self.experiment_name.copy()
        for metric in metrics:
            result_train[metric] = metrics_train[metric].tolist()
            result_val[metric] = metrics_val[metric].tolist()
        return result_train, result_val


@dataclass
class CVBenchmark(SimpleBenchmark):
    """
    example use:
    >>> benchmark_cv = CVBenchmark(
    >>>     experiments = experiments_cv,
    >>>     metrics = ['MAE', 'MSE'])
    >>> results_summary, results_train, results_val = benchmark_cv.run()
    """

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
