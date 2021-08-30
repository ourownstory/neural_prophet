import time
from collections import OrderedDict
from attrdict import AttrDict
from neuralprophet.forecaster import NeuralProphet
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch import optim
import logging
from tqdm import tqdm

from neuralprophet import configure
from neuralprophet import time_net
from neuralprophet import time_dataset
from neuralprophet import df_utils
from neuralprophet import utils
from neuralprophet import utils_torch
from neuralprophet.plot_forecast import plot, plot_components
from neuralprophet.plot_model_parameters import plot_parameters
from neuralprophet import metrics
from neuralprophet.utils import set_logger_level

log = logging.getLogger("NP.forecaster")

class Classification_NP(NeuralProphet):
    """NeuralProphet binary classifier.
    A simple classifier for binary classes time-series. One can notice that n_lags is 
    set to 0 becasue y is the output column. A lagged-regressor is required so the classification
    can be accomplished.
    """
    def __init__(self, 
        growth='linear',
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
        impute_missing=True):
        super().__init__(growth,
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
        impute_missing)
        kwargs = locals()
        self.config_train = configure.from_kwargs(configure.Train, kwargs) 
        self.metrics = metrics.MetricsCollection(
                metrics=[
                    metrics.LossMetric(self.config_train.loss_func),
                    metrics.Accuracy(),
                    metrics.Balanced_Accuracy(),
                    metrics.F1Score()
                ],
                value_metrics=[
                    # metrics.ValueMetric("Loss"),
                    # metrics.ValueMetric("RegLoss"),
                ],
            ) 
    def fit(self, df, freq, epochs, validate_each_epoch, valid_p, progress_bar, plot_live_loss):
        if self.n_lags>0:
            log.warning('Warning! Auto-regression is activated, the model is using the classifier label as input. Please consider setting n_lags=0.')
        elif self.n_lags==0 and self.n_regressors==0:
            log.warning('Warning! Please add lagged regressor as the input of the classifier')
        return super().fit(df, freq, epochs=epochs, validate_each_epoch=validate_each_epoch, valid_p=valid_p, progress_bar=progress_bar, plot_live_loss=plot_live_loss)

    def predict(self, df):
        df = super().predict(df)
        def sigmoid(x):
            y=1/(1+np.math.e**(-1*x))
            return y
        # create a line for each forecast_lag
        # 'yhat<i>' is the forecast for 'y' at 'ds' from i steps ago.

        for i in range(self.n_forecasts):
            yhat=df["yhat{}".format(i + 1)]
            df["yhat{}".format(i + 1)] = sigmoid(yhat)
            df["residual{}".format(i + 1)] = sigmoid(yhat) - df["y"]
        return df

        