import logging

import torchmetrics

log = logging.getLogger("NP.metrics")

METRICS = {
    "MAE": torchmetrics.MeanAbsoluteError(),
    "MSE": torchmetrics.MeanSquaredError(squared=True),
    "RMSE": torchmetrics.MeanSquaredError(squared=False),
}


def get_metrics(metric_input):
    """
    Returns a list of metrics.

    Parameters
    ----------
        metrics : input received from the user
            List of metrics to use.

    Returns
    -------
        dict
            Dict of torchmetrics.Metric metrics.
    """
    if metric_input is None:
        return {}
    elif metric_input is True:
        return {k: v for k, v in METRICS.items() if k in ["MAE", "RMSE"]}
    elif isinstance(metric_input, str):
        if metric_input.upper() in METRICS.keys():
            return {metric_input: METRICS[metric_input]}
        else:
            raise ValueError("Received unsupported argument for collect_metrics.")
    elif isinstance(metric_input, list):
        if all([m.upper() in METRICS.keys() for m in metric_input]):
            return {k: v for k, v in METRICS.items() if k in metric_input}
        else:
            raise ValueError("Received unsupported argument for collect_metrics.")
    elif isinstance(metric_input, dict):
        if all([isinstance(_metric, torchmetrics.Metric) for _, _metric in metric_input.items()]):
            return metric_input
        else:
            raise ValueError(
                "Received unsupported argument for collect_metrics. All metrics must be an instance of torchmetrics.Metric."
            )
    elif metric_input is not False:
        raise ValueError("Received unsupported argument for collect_metrics.")
