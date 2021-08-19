import pandas as pd
import numpy as np
import torch

from neuralprophet import metrics
    
class Classification():
    """
        NeuralProphet Classification
    """
    def __init__(
        self,
        loss_func=torch.nn.BCEWithLogitsLoss(),
    ):
        self.loss_func = loss_func
        
    def set_metrics(loss_function=torch.nn.BCEWithLogitsLoss()):
        metric = metrics.MetricsCollection(
            metrics=[
                metrics.LossMetric(loss_function),
                metrics.Accuracy(),
                metrics.Balanced_Accuracy(),
                metrics.F1Score(),
            ],
            value_metrics=[]
        )
        return metric