import numpy as np
import pandas as pd
import torch
from attrdict import AttrDict
from collections import OrderedDict
import hdays as hdays_part2
import holidays as hdays_part1
import warnings

def get_regularization_lambda(sparsity, lambda_delay_epochs=None, epoch=None):
    """Computes regularization lambda strength for a given sparsity and epoch.

    Args:
        sparsity (float): (0, 1] how dense the weights shall be.
            Smaller values equate to stronger regularization
        lambda_delay_epochs (int): how many epochs to wait bbefore adding full regularization
        epoch (int): current epoch number

    Returns:
        lam (float): regularization strength
    """
    if sparsity is not None and sparsity < 1:
        lam = 0.02 * (1.0 / sparsity - 1.0)
        if lambda_delay_epochs is not None and epoch < lambda_delay_epochs:
            lam = lam * epoch / (1.0 * lambda_delay_epochs)
            # lam = lam * (epoch / (1.0 * lambda_delay_epochs))**2
    else:
        lam = None
    return lam


def reg_func_ar(weights):
    """Regularization of coefficients based on AR-Net paper

    Args:
        weights (torch tensor): Model weights to be regularized towards zero

    Returns:
        regularization loss, scalar

    """
    # abs_weights = torch.abs(weights)
    abs_weights = torch.abs(weights.clone())
    reg = torch.div(2.0, 1.0 + torch.exp(-3*(1e-12+abs_weights).pow(1/3.0))) - 1.0
    reg = torch.mean(reg).squeeze()
    return reg

def reg_func_trend(weights, threshold=None):
    """Regularization of weights to induce sparcity

    Args:
        weights (torch tensor): Model weights to be regularized towards zero
        threshold (float): value below which not to regularize weights

    Returns:
        regularization loss, scalar
    """
    abs_weights = torch.abs(weights.clone())
    if threshold is not None:
        abs_weights = torch.clamp(abs_weights - threshold, min=0.0)
    # reg = 10*torch.div(2.0, 1.0 + torch.exp(-2*(1e-12+abs_weights/10).pow(0.5))) - 1.0
    # reg = (1e-12+abs_weights).pow(0.5)
    reg = abs_weights  # Most stable
    reg = torch.sum(reg).squeeze()
    return reg


def reg_func_season(weights):
    """Regularization of weights to induce sparcity

    Args:
        weights (torch tensor): Model weights to be regularized towards zero

    Returns:
        regularization loss, scalar
    """
    abs_weights = torch.abs(weights.clone())
    # reg = torch.div(2.0, 1.0 + torch.exp(-2*(1e-9+abs_weights).pow(0.5))) - 1.0
    # reg = (1e-12+abs_weights).pow(0.5)
    reg = abs_weights  # Most stable
    reg = torch.mean(reg).squeeze()
    return reg

def reg_func_holidays(weights, events_dim, events_config=None, country_holidays_config=None):
    """Regularization of weights to induce sparcity

    Args:
        weights (torch tensor): Model weights to be regularized towards zero

    Returns:
        regularization loss, scalar
    """
    abs_weights = torch.abs(weights.clone())
    # reg = torch.div(2.0, 1.0 + torch.exp(-2*(1e-9+abs_weights).pow(0.5))) - 1.0
    # reg = (1e-12+abs_weights).pow(0.5)

    # if events_config is not None:
    #     for event, configs in events_config.items():
    #         reg_lambda = configs.reg_lambda
    #         rows = events_dim[events_dim['event'] == event]
    #         start_loc = rows.index.min()
    #         end_loc = rows.index.max() + 1
    #         weights = abs_weights[start_loc:end_loc]
    #         reg_weights =
    reg = abs_weights  # Most stable
    reg = torch.mean(reg).squeeze()
    return reg


def symmetric_total_percentage_error(values, estimates):
    """ Compute STPE

    Args:
        values (np.array):
        estimates (np.array):

    Returns:
        scalar (float)
    """
    sum_abs_diff = np.sum(np.abs(estimates - values))
    sum_abs = np.sum(np.abs(estimates) + np.abs(values))
    return 100 * sum_abs_diff / (10e-9 + sum_abs)


def season_config_to_model_dims(season_config):
    """Convert the NeuralProphet seasonal model configuration to input dims for TimeNet model.

    Args:
        season_config (AttrDict): NeuralProphet seasonal model configuration

    Returns:
        seasonal_dims (dict(int)): input dims for TimeNet model
    """
    if season_config is None or len(season_config.periods) < 1:
        return None
    seasonal_dims = OrderedDict({})
    for name, period in season_config.periods.items():
        resolution = period['resolution']
        if season_config.type == 'fourier':
            resolution = 2 * resolution
        seasonal_dims[name] = resolution
    return seasonal_dims

def get_holidays_from_country(country, dates=None):
    """
    Return all possible holiday names of given country

    Args:
        country (string): country name to retrieve country specific holidays
        dates (pd.Series): datestamps

    Returns:
        A set of all possible holiday names of given country
    """

    if dates is None:
        years = np.arange(1995, 2045)
    else:
        years = list({x.year for x in dates})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            holiday_names = getattr(hdays_part2, country)(years=years).values()
    except AttributeError:
        try:
            holiday_names = getattr(hdays_part1, country)(years=years).values()
        except AttributeError:
            raise AttributeError(
                "Holidays in {} are not currently supported!".format(country))
    return set(holiday_names)

def events_config_to_model_dims(events_config, country_holidays_config):
    """
    Convert the NeuralProphet user specified events configurations along with country specific
        holidays to input dims for TimeNet model.
    Args:
        events_config (OrderedDict): Configurations (upper, lower windows, regularization) for user specified events
        country_holidays_config (OrderedDict): Configurations (holiday_names, upper, lower windows, regularization)
            for country specific holidays

    Returns:
        events_dims (pd.DataFrame): Dataframe with columns 'event' which denotes the event name and
            'event_delim', a unique identifier for each offset of every event. The dataframe rows are
            sorted based on 'event_delim'
    """
    events_dims = pd.DataFrame(columns=['event', 'event_delim'])
    if events_config is not None:
        for event, configs in events_config.items():
            for offset in range(configs.lower_window, configs.upper_window + 1):
                event_delim = create_event_names_for_offsets(event, offset)
                events_dims = events_dims.append({'event': event, 'event_delim': event_delim}, ignore_index=True)

    if country_holidays_config is not None:
        lower_window = country_holidays_config["lower_window"]
        upper_window = country_holidays_config["upper_window"]
        for country_holiday in country_holidays_config["holiday_names"]:
            for offset in range(lower_window, upper_window + 1):
                holiday_delim = create_event_names_for_offsets(country_holiday, offset)
                events_dims = events_dims.append({'event': country_holiday, 'event_delim': holiday_delim}, ignore_index=True)

    # sort based on event_delim
    events_dims = events_dims.sort_values(by='event_delim').reset_index(drop=True)
    return events_dims

def create_event_names_for_offsets(event_name, offset):
    """
    Create names for offsets of every event
    Args:
        event_name (string): Name of the event
        offset (int): Offset of the event

    Returns:
        offset_name (string): A name created for the offset of the event
    """
    offset_name = '{}_delim_{}{}'.format(
        event_name,
        '+' if offset >= 0 else '-',
        abs(offset)
    )
    return offset_name

def set_auto_seasonalities(dates, season_config, verbose=False):
    """Set seasonalities that were left on auto or set by user.

    Turns on yearly seasonality if there is >=2 years of history.
    Turns on weekly seasonality if there is >=2 weeks of history, and the
    spacing between dates in the history is <7 days.
    Turns on daily seasonality if there is >=2 days of history, and the
    spacing between dates in the history is <1 day.

    Args:
        dates (pd.Series): datestamps
        season_config (AttrDict): NeuralProphet seasonal model configuration, as after __init__
        verbose (bool):

    Returns:
        season_config (AttrDict): processed NeuralProphet seasonal model configuration

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
    season_config = season_config if len(season_config.periods) > 0 else None
    return season_config

def print_epoch_metrics(metrics, val_metrics=None, e=0):
    if val_metrics is not None and len(val_metrics) > 0:
        val = OrderedDict({"{}_val".format(key): value for key, value in val_metrics.items()})
        metrics = {**metrics, **val}
    metrics_df = pd.DataFrame({**metrics,}, index=[e + 1])
    metrics_string = metrics_df.to_string(float_format=lambda x: "{:6.3f}".format(x))
    if e > 0: metrics_string = metrics_string.splitlines()[1]
    print(metrics_string)

