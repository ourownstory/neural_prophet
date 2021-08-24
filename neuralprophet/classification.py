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

    
class Classification_NP(NeuralProphet):
    def __init__(self,
    n_lags=0,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    loss_func='bce',
    learning_rate=None,
    ar_sparsity=None,
    optimizer="AdamW",
    epochs=None,
    batch_size=None,
    train_speed=None
    ):
        kwargs = locals()
        super(Classification_NP,self).__init__()  
        self.classifier_flag=True   
        self.n_lags=n_lags   
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
                    metrics.ValueMetric("RegLoss"),
                ],
            ) 
        self.season_config = configure.AllSeason(
            yearly_arg=yearly_seasonality,
            weekly_arg=weekly_seasonality,
            daily_arg=daily_seasonality,
        )
        