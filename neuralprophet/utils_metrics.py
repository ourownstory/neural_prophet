import logging

import torchmetrics

log = logging.getLogger("NP.metrics")

METRICS = {
    # "short_name": [torchmetrics.Metric name, {optional args}]
    "MAE": ["MeanAbsoluteError", {}],
    "MSE": ["MeanSquaredError", {"squared": True}],
    "RMSE": ["MeanSquaredError", {"squared": False}],
}


def get_metrics(metric_input):
    """
    Returns a dict of metrics.

    Parameters
    ----------
        metrics : input received from the user
            List of metrics to use.

    Returns
    -------
        dict
            Dict of names of torchmetrics.Metric metrics
    """
    if metric_input is None:
        return {}
    elif metric_input is True:
        return {"MAE": METRICS["MAE"], "RMSE": METRICS["RMSE"]}
    elif isinstance(metric_input, str):
        if metric_input.upper() in METRICS.keys():
            return {metric_input: METRICS[metric_input.upper()]}
        else:
            raise ValueError("Received unsupported argument for collect_metrics.")
    elif isinstance(metric_input, list):
        if all([m.upper() in METRICS.keys() for m in metric_input]):
            return {m: METRICS[m.upper()] for m in metric_input}
        else:
            raise ValueError("Received unsupported argument for collect_metrics.")
    elif isinstance(metric_input, dict):
        # check if all values are names belonging to torchmetrics.Metric
        try:
            for _metric in metric_input.values():
                torchmetrics.__dict__[_metric]()
        except KeyError:
            raise ValueError(
                "Received unsupported argument for collect_metrics."
                "All metrics must be valid names of torchmetrics.Metric objects."
            )
        return {k: [v, {}] for k, v in metric_input.items()}
    elif metric_input is not False:
        raise ValueError("Received unsupported argument for collect_metrics.")
