import time
from collections import OrderedDict
import numpy as np
import pandas as pd
from neuralprophet.forecaster import NeuralProphet
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

from neuralprophet import configure
from neuralprophet import time_net
from neuralprophet import time_dataset
from neuralprophet import df_utils
from neuralprophet import utils
from neuralprophet.plot_forecast import plot, plot_components
from neuralprophet.plot_model_parameters import plot_parameters
from neuralprophet import metrics
from neuralprophet.df_utils import get_max_num_lags

from neuralprophet.utils import set_logger_level


log = logging.getLogger("NP.forecaster")


class Classification_NP(NeuralProphet):
    """NeuralProphet binary classifier.
    A simple classifier for binary classes time-series. One can notice that n_lags is
    set to 0 becasue y is the output column. A lagged regressor is required so the classification
    can be accomplished.
    """

    # def __init__(self, **kwargs):
    #     super(Classification_NP, self).__init__(**kwargs)

    def __init__(
        self,
        growth="linear",
        changepoints=None,
        n_changepoints=10,
        changepoints_range=0.9,
        trend_reg=0,
        trend_reg_threshold=False,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        seasonality_mode="additive",
        seasonality_reg=0,
        n_forecasts=1,
        n_lags=0,
        num_hidden_layers=0,
        d_hidden=None,
        ar_reg=None,
        learning_rate=None,
        epochs=None,
        batch_size=None,
        loss_func="bce",
        optimizer="AdamW",
        newer_samples_weight=2,
        newer_samples_start=0.0,
        impute_missing=True,
        collect_metrics=True,
        normalize="auto",
        global_normalization=False,
        global_time_normalization=True,
        unknown_data_normalization=False,
    ):
        super().__init__(
            growth,
            changepoints,
            n_changepoints,
            changepoints_range,
            trend_reg,
            trend_reg_threshold,
            yearly_seasonality,
            weekly_seasonality,
            daily_seasonality,
            seasonality_mode,
            seasonality_reg,
            n_forecasts,
            n_lags,
            num_hidden_layers,
            d_hidden,
            ar_reg,
            learning_rate,
            epochs,
            batch_size,
            loss_func,
            optimizer,
            newer_samples_weight,
            newer_samples_start,
            impute_missing,
            collect_metrics,
            normalize,
            global_normalization,
            global_time_normalization,
            unknown_data_normalization,
        )
        kwargs = locals()
        self.classification_task = True

        METRICS = {
            "acc": metrics.Accuracy,
            "bal_acc": metrics.Balanced_Accuracy,
            "f1": metrics.F1Score,
        }
        # General
        self.name = "NeuralProphetBinaryClassifier"
        self.config_train = configure.from_kwargs(configure.Train, kwargs)
        self.loss_func_name = loss_func
        if collect_metrics is None:
            collect_metrics = []
        elif collect_metrics is True:
            collect_metrics = ["acc", "bal_acc", "f1"]
        elif isinstance(collect_metrics, str):
            if not collect_metrics.lower() in METRICS.keys():
                raise ValueError("Received unsupported argument for collect_metrics.")
            collect_metrics = [collect_metrics]
        elif isinstance(collect_metrics, list):
            if not all([m.lower() in METRICS.keys() for m in collect_metrics]):
                raise ValueError("Received unsupported argument for collect_metrics.")
        elif collect_metrics is not False:
            raise ValueError("Received unsupported argument for collect_metrics.")

        self.metrics = None
        if isinstance(collect_metrics, list):
            self.metrics = metrics.MetricsCollection(
                metrics=[metrics.LossMetric(self.config_train.loss_func)]
                + [METRICS[m.lower()]() for m in collect_metrics],
                value_metrics=[metrics.ValueMetric("Loss")],
            )

    def fit(
        self,
        df,
        freq="auto",
        validation_df=None,
        progress="bar",
        minimal=False,
    ):
        max_lags = get_max_num_lags(self.config_covar, self.n_lags)
        if self.n_lags > 0:
            log.warning(
                "Warning! Auto-regression is activated, the model is using the classifier label as input. Please consider setting n_lags=0."
            )
        if max_lags == 0:
            log.warning("Warning! Please add lagged regressor as the input of the classifier")
        if self.loss_func_name in ["bce", "bceloss"]:
            log.info("Classification with bce loss")
        else:
            raise NotImplementedError(
                "Currently NeuralProphet does not support {} loss function. Please, set loss function to 'bce' ".format(
                    self.loss_func_name
                )
            )
        return super().fit(
            df,
            freq=freq,
            validation_df=validation_df,
            progress=progress,
            minimal=minimal,
        )

    def predict(self, df):
        df = super().predict(df)
        # create a line for each forecast_lag
        # 'yhat<i>' is the forecast for 'y' at 'ds' from i steps ago (value between 0 and 1).
        df_dict, received_unnamed_df = df_utils.prep_copy_df_dict(df)
        for key, df_i in df_dict.items():
            for i in range(self.n_forecasts):
                df_i = df_i.rename(columns={"yhat{}".format(i + 1): "yhat_raw{}".format(i + 1)}).copy(deep=True)
                yhat = df_i["yhat_raw{}".format(i + 1)]
                yhat = np.array(yhat.values, dtype=np.float64)
                df_i["yhat{}".format(i + 1)] = torch.gt(torch.tensor(yhat), 0.5).numpy()
                df_i["residual{}".format(i + 1)] = df_i["yhat_raw{}".format(i + 1)] - df_i["y"]
                df_dict[key] = df_i
        df = df_utils.maybe_get_single_df_from_df_dict(df_dict, received_unnamed_df)
        return df
