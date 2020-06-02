import numpy as np
import pandas as pd
import torch
from attrdict import AttrDict
from collections import OrderedDict
from datetime import timedelta, datetime

from copy import copy, deepcopy

def get_regularization_lambda(sparsity, lambda_delay_epochs=None, epoch=None):
    if sparsity is not None and sparsity < 1:
        lam = 0.02 * (1.0 / sparsity - 1.0)
        if lambda_delay_epochs is not None and epoch < lambda_delay_epochs:
            lam = lam * epoch / (1.0 * lambda_delay_epochs)
            # lam = lam * (epoch / (1.0 * lambda_delay_epochs))**2
    else:
        lam = None
    return lam


def regulariziation_function_ar(weights):
    # abs_weights = torch.abs(weights)
    abs_weights = torch.abs(weights.clone())
    abs_weights = torch.sum(abs_weights, dim=0)
    reg = torch.div(2.0, 1.0 + torch.exp(-3.0 * abs_weights.pow(1.0 / 3.0))) - 1.0
    return reg

def regulariziation_function_trend(weights, threshold=None):
    abs_weights = torch.abs(weights)
    if threshold is not None:
        abs_weights = torch.clamp(abs_weights - threshold, min=0.0)
    # reg = 10*regulariziation_function_ar(abs_weights)
    reg = torch.abs(abs_weights)
    return reg


def symmetric_total_percentage_error(values, estimates):
    sum_abs_diff = np.sum(np.abs(estimates - values))
    sum_abs = np.sum(np.abs(estimates) + np.abs(values))
    return 100 * sum_abs_diff / (10e-9 + sum_abs)


def make_future_dataframe(history_dates, periods, freq='D', include_history=True):
    """Simulate the trend using the extrapolated generative model.

    Parameters
    ----------
    periods: Int number of periods to forecast forward.
    freq: Any valid frequency for pd.date_range, such as 'D' or 'M'.
    include_history: Boolean to include the historical dates in the data
        frame for predictions.

    Returns
    -------
    pd.Dataframe that extends forward from the end of self.history for the
    requested number of periods.
    """
    if history_dates is None:
        raise Exception('Model has not been fit.')
    last_date = history_dates.max()
    dates = pd.date_range(
        start=last_date,
        periods=periods + 1,  # An extra in case we include start
        freq=freq)
    dates = dates[dates > last_date]  # Drop start if equals last_date
    dates = dates[:periods]  # Return correct number of periods

    if include_history:
        dates = np.concatenate((np.array(history_dates), dates))

    return pd.DataFrame({'ds': dates})



def fourier_series(dates, period, series_order):
    """Provides Fourier series components with the specified frequency
    and order.

    Parameters
    ----------
    dates: pd.Series containing timestamps.
    period: Number of days of the period.
    series_order: Number of components.

    Returns
    -------
    Matrix with seasonality features.
    """
    # convert to days since epoch
    t = np.array(
        (dates - datetime(1970, 1, 1))
            .dt.total_seconds()
            .astype(np.float)
    ) / (3600 * 24.)
    features = np.column_stack(
        [fun((2.0 * (i + 1 ) * np.pi * t / period))
         for i in range(series_order)
         for fun in (np.sin, np.cos)
         ])
    return features


def seasonal_features_from_dates(dates, season_config):
    """Dataframe with seasonality features.

    Includes seasonality features, holiday features, and added regressors.

    Parameters
    ----------
    dates: pd.Series with dates for computing seasonality features and any
        added regressors.

    Returns
    -------
    Dictionary with keys 'additive' and 'multiplicative' and subkeys for each
        period name containing an np.array with the respective regression features.
    """
    assert len(dates.shape) == 1
    seasonalities = OrderedDict({})
    # Seasonality features
    for name, period in season_config.periods.items():
        if period['resolution'] > 0:
            features = fourier_series(
                dates=dates,
                period=period['period'],
                series_order=period['resolution'],
            )
            seasonalities[name] = features
    return seasonalities


def apply_fun_to_seasonal_dict(seasonalities, fun):
    for mode, seasons_in in seasonalities.items():
        for name, values in seasons_in.items():
            seasonalities[mode][name] = fun(values)
    return seasonalities


def apply_fun_to_seasonal_dict_copy(seasonalities, fun):
    out = AttrDict({})
    for mode, seasons_in in seasonalities.items():
        out[mode] = OrderedDict({})
        for name, values in seasons_in.items():
            out[mode][name] = fun(values)
    return out


def season_config_to_model_dims(season_config):
    if len(season_config.periods) < 1:
        return None
    seasonal_dims = OrderedDict({})
    for name, period in season_config.periods.items():
        resolution = period['resolution']
        if season_config.type == 'fourier':
            resolution = 2 * resolution
        seasonal_dims[name] = resolution
    return seasonal_dims


def set_auto_seasonalities(dates, season_config, verbose=False):
    """Set seasonalities that were left on auto.

    Turns on yearly seasonality if there is >=2 years of history.
    Turns on weekly seasonality if there is >=2 weeks of history, and the
    spacing between dates in the history is <7 days.
    Turns on daily seasonality if there is >=2 days of history, and the
    spacing between dates in the history is <1 day.
    """
    first = dates.min()
    last = dates.max()
    dt = dates.diff()
    min_dt = dt.iloc[dt.values.nonzero()[0]].min()
    auto_disable = {
        "yearly": last - first < pd.Timedelta(days=730),
        "weekly": ((last - first < pd.Timedelta(weeks=2)) or (min_dt >= pd.Timedelta(weeks=1))),
        "daily": ((last - first < pd.Timedelta(days=2)) or (min_dt >= pd.Timedelta(days=1))),
    }
    for name, period in season_config.periods.items():
        arg = period.arg
        default_resolution = period.resolution
        if arg == 'auto':
            resolution = 0
            if auto_disable[name]:
                # logger.info(
                print(
                    'Disabling {name} seasonality. Run prophet with '
                    '{name}_seasonality=True to override this.'
                    .format(name=name)
                )
            else:
                resolution = default_resolution
        elif arg is True:
            resolution = default_resolution
        elif arg is False:
            resolution = 0
        else:
            resolution = int(arg)
        season_config.periods[name].resolution = resolution

    new_periods = OrderedDict({})
    for name, period in season_config.periods.items():
        if period.resolution > 0:
            new_periods[name] = period
    season_config.periods = new_periods
    if verbose:
        print(season_config)
    return season_config if len(season_config.periods) > 0 else None
