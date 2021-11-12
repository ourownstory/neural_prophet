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

from neuralprophet.utils import set_logger_level

log = logging.getLogger("NP.forecaster")
print("Neural Prophet Classification running")


class Classification_NP(NeuralProphet):
    """NeuralProphet binary classifier.
    A simple classifier for binary classes time-series. One can notice that n_lags is
    set to 0 becasue y is the output column. A lagged-regressor is required so the classification
    can be accomplished.
    """

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
        ar_sparsity=None,
        learning_rate=None,
        epochs=None,
        batch_size=None,
        loss_func="bce",
        optimizer="AdamW",
        train_speed=None,
        normalize="auto",
        impute_missing=True,
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
            ar_sparsity,
            learning_rate,
            epochs,
            batch_size,
            loss_func,
            optimizer,
            train_speed,
            normalize,
            impute_missing,
        )
        kwargs = locals()
        self.config_train = configure.from_kwargs(configure.Train, kwargs)
        self.metrics = metrics.MetricsCollection(
            metrics=[
                metrics.LossMetric(self.config_train.loss_func),
                metrics.Accuracy(),
                metrics.Balanced_Accuracy(),
                metrics.F1Score(),
            ],
            value_metrics=[
                # metrics.ValueMetric("Loss"),
                # metrics.ValueMetric("RegLoss"),
            ],
        )

    def fit(
        self,
        df,
        freq,
        validation_df=None,
        epochs=None,
        local_modeling=False,
        progress_bar=True,
        plot_live_loss=False,
        progress_print=True,
        minimal=False,
    ):

        if self.n_lags > 0:
            log.warning(
                "Warning! Auto-regression is activated, the model is using the classifier label as input. Please consider setting n_lags=0."
            )
        elif self.n_lags == 0 and self.n_regressors == 0:
            log.warning("Warning! Please add lagged regressor as the input of the classifier")
        return super().fit(
            df,
            freq,
            validation_df=validation_df,
            epochs=epochs,
            local_modeling=local_modeling,
            progress_bar=progress_bar,
            plot_live_loss=plot_live_loss,
            progress_print=progress_print,
            minimal=minimal,
        )

    def predict(self, df):
        df = super().predict(df)
        # create a line for each forecast_lag
        # 'yhat<i>' is the forecast for 'y' at 'ds' from i steps ago.

        for i in range(self.n_forecasts):
            yhat = df["yhat{}".format(i + 1)]
            yhat = np.array(yhat.values, dtype=np.float64)
            df["yhat{}".format(i + 1)] = torch.sigmoid(torch.tensor(yhat)).numpy()
            df["residual{}".format(i + 1)] = torch.sigmoid(torch.tensor(yhat)).numpy() - df["y"]
        return df
