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

def reg_func_holidays(weights):
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

def holiday_config_to_model_dims(holidays_config, country_holidays_set):
    """Convert the NeuralProphet seasonal model configuration to input dims for TimeNet model.

    Args:
        season_config (AttrDict): NeuralProphet seasonal model configuration

    Returns:
        seasonal_dims (dict(int)): input dims for TimeNet model
    """
    holidays_dims = pd.DataFrame(columns=['holiday', 'holiday_delim'])
    if holidays_config is not None:
        for holiday, windows in holidays_config.items():
            for offset in range (windows[0], windows[1] + 1):
                holiday_delim = create_holiday_names_for_offsets(holiday, offset)
                holidays_dims = holidays_dims.append({'holiday': holiday, 'holiday_delim': holiday_delim}, ignore_index=True)
            # holidays_dims = abs(windows[0]) + abs(windows[1]) + 1

    if country_holidays_set is not None:
        for country_holiday in country_holidays_set:
            holiday_delim = create_holiday_names_for_offsets(country_holiday, 0)
            holidays_dims = holidays_dims.append({'holiday': country_holiday, 'holiday_delim': holiday_delim}, ignore_index=True)
            # holidays_dims[country_holiday] = 1

    # sort based on holiday_delim
    holidays_dims = holidays_dims.sort_values(by='holiday_delim').reset_index(drop=True)
    return holidays_dims

def create_holiday_names_for_offsets(holiday_name, offset):
    offset_name = '{}_delim_{}{}'.format(
        holiday_name,
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


def get_holiday_names(country):
    """Return all possible holiday names of given country

    Parameters
    ----------
    country: country name

    Returns
    -------
    A set of all possible holiday names of given country
    """
    years = np.arange(1995, 2045)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            holiday_names = getattr(hdays_part2, country)(years=years).values()
    except AttributeError:
        try:
            holiday_names = getattr(hdays_part1, country)(years=years).values()
        except AttributeError:
            raise AttributeError(
                "Holidays in {} are not currently    supported!".format(country))
    return set(holiday_names)

def set_holiday_configs(dates, user_specified_holiday_config=None, country=None):
    """Make dataframe of holidays for given years and countries

    Parameters
    ----------
    year_list: a list of years
    country: country name

    Returns
    -------
    Dataframe with 'ds' and 'holiday', which can directly feed
    to 'holidays' params in Prophet
    """
    n_country_holiday_params = 0
    n_holiday_parameters = 0
    country_holidays = None
    train_holiday_names = []

    if country is not None:
        year_list = list({x.year for x in dates})

        try:
            country_holidays = getattr(hdays_part2, country)(years=year_list)
        except AttributeError:
            try:
                country_holidays = getattr(hdays_part1, country)(years=year_list)
            except AttributeError:
                raise AttributeError(
                    "Holidays in {} are not currently supported!".format(country))

        country_holidays = set(country_holidays.values())
        n_country_holiday_params = len(country_holidays)
        # # update the holiday config
        # if user_specified_holiday_config is None:
        #     user_specified_holiday_config = dict({})
        # for country_holiday in country_holidays:
        #     if country_holiday not in user_specified_holiday_config.keys():
        #         user_specified_holiday_config[country_holiday] = (0, 0)

    # calculate the total number of holiday parameters
    if user_specified_holiday_config is not None:
        n_holiday_parameters += sum([(abs(i[0]) + i[1] + 1) for i in list(user_specified_holiday_config.values())])
        train_holiday_names = list(user_specified_holiday_config.keys())
    if country is not None:
        n_holiday_parameters += n_country_holiday_params
        train_holiday_names += list(country_holidays)
    # country_specific_holidays_df = pd.DataFrame(list(country_holidays.items()),
    #                                             columns=['ds', 'holiday'])
    # country_specific_holidays_df.reset_index(inplace=True, drop=True)
    # country_specific_holidays_df['ds'] = pd.to_datetime(country_specific_holidays_df['ds'])
    return country_holidays, train_holiday_names, n_holiday_parameters


def print_epoch_metrics(metrics, val_metrics=None, e=0):
    if val_metrics is not None and len(val_metrics) > 0:
        val = OrderedDict({"{}_val".format(key): value for key, value in val_metrics.items()})
        metrics = {**metrics, **val}
    metrics_df = pd.DataFrame({**metrics,}, index=[e + 1])
    metrics_string = metrics_df.to_string(float_format=lambda x: "{:6.3f}".format(x))
    if e > 0: metrics_string = metrics_string.splitlines()[1]
    print(metrics_string)

