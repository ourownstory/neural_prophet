import os
import sys
import math
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
from neuralprophet import hdays as hdays_part2
import holidays as pyholidays
import warnings
import logging

log = logging.getLogger("NP.utils")


def save(model, path):
    """save a fitted np model to a disk file.

    Parameters
    ----------
        model : np.forecaster.NeuralProphet
            input model that is fitted
        path : str
            path and filename to be saved. filename could be any but suggested to have extension .np.
    Examples
    --------
    After you fitted a model, you may save the model to save_test_model.np
        >>> from neuralprophet import save
        >>> save(model, "test_save_model.np")
    """
    torch.save(model, path)


def load(path):
    """retrieve a fitted model from a .np file that was saved by save.

    Parameters
    ----------
        path : str
            path and filename to be saved. filename could be any but suggested to have extension .np.

    Returns
    -------
        np.forecaster.NeuralProphet
            previously saved model

    Examples
    --------
    Saved model could be loaded from disk file test_save_model.np
        >>> from neuralprophet import load
        >>> model = load("test_save_model.np")
    """
    return torch.load(path)


def reg_func_abs(weights):
    """Regularization of weights to induce sparcity

    Parameters
    ----------
        weights : torch.Tensor
            Model weights to be regularized towards zero

    Returns
    -------
        torch.Tensor
            Regularization loss
    """
    return torch.mean(torch.abs(weights)).squeeze()


def reg_func_trend(weights, threshold=None):
    """Regularization of weights to induce sparcity

    Parameters
    ----------
        weights : torch.Tensor
            Model weights to be regularized towards zero
        threshold : float
            Value below which not to regularize weights

    Returns
    -------
        torch.Tensor
            regularization loss
    """
    abs_weights = torch.abs(weights)
    if threshold is not None and not math.isclose(threshold, 0):
        abs_weights = torch.clamp(abs_weights - threshold, min=0.0)
    reg = torch.mean(torch.sum(abs_weights, dim=1)).squeeze()
    return reg


def reg_func_season(weights):
    return reg_func_abs(weights)


def reg_func_events(events_config, country_holidays_config, model):
    """
    Regularization of events coefficients to induce sparcity

    Parameters
    ----------
        events_config : OrderedDict
            Configurations (upper, lower windows, regularization) for user specified events
        country_holidays_config : OrderedDict
            Configurations (holiday_names, upper, lower windows, regularization)
            for country specific holidays
        model : TimeNet
            The TimeNet model object

    Returns
    -------
        scalar
            Regularization loss
    """
    reg_events_loss = 0.0
    if events_config is not None:
        for event, configs in events_config.items():
            reg_lambda = configs.reg_lambda
            if reg_lambda is not None:
                weights = model.get_event_weights(event)
                for offset in weights.keys():
                    reg_events_loss += reg_lambda * reg_func_abs(weights[offset])

    if country_holidays_config is not None:
        reg_lambda = country_holidays_config.reg_lambda
        if reg_lambda is not None:
            for holiday in country_holidays_config.holiday_names:
                weights = model.get_event_weights(holiday)
                for offset in weights.keys():
                    reg_events_loss += reg_lambda * reg_func_abs(weights[offset])
    return reg_events_loss


def reg_func_regressors(regressors_config, model):
    """
    Regularization of regressors coefficients to induce sparsity

    Parameters
    ----------
        regressors_config : OrderedDict
            Configurations for user specified regressors
        model : TimeNet
            TimeNet model object

    Returns
    -------
        scalar
            Regularization loss
    """
    reg_regressor_loss = 0.0
    for regressor, configs in regressors_config.items():
        reg_lambda = configs.reg_lambda
        if reg_lambda is not None:
            weight = model.get_reg_weights(regressor)
            reg_regressor_loss += reg_lambda * reg_func_abs(weight)

    return reg_regressor_loss


def symmetric_total_percentage_error(values, estimates):
    """Compute STPE

    Parameters
    ----------
        values : np.array
            Input values
        estimates : np.array
            Respective estimates of input values

    Returns
    -------
        float
            Symmetric total percentage error
    """
    sum_abs_diff = np.sum(np.abs(estimates - values))
    sum_abs = np.sum(np.abs(estimates) + np.abs(values))
    return 100 * sum_abs_diff / (10e-9 + sum_abs)


def season_config_to_model_dims(season_config):
    """Convert the NeuralProphet seasonal model configuration to input dims for TimeNet model.

    Parameters
    ----------
        season_config : configure.AllSeason
            NeuralProphet seasonal model configuration

    Returns
    -------
        dict(int)
            Input dims for TimeNet model
    """
    if season_config is None or len(season_config.periods) < 1:
        return None
    seasonal_dims = OrderedDict({})
    for name, period in season_config.periods.items():
        resolution = period.resolution
        if season_config.computation == "fourier":
            resolution = 2 * resolution
        seasonal_dims[name] = resolution
    return seasonal_dims


def get_holidays_from_country(country, df=None):
    """
    Return all possible holiday names of given country

    Parameters
    ----------
        country : string
            Country name to retrieve country specific holidays
        df : Dataframe
            Dataframe from which datestamps will be retrieved from

    Returns
    -------
        set
            All possible holiday names of given country
    """
    if df is None:
        years = np.arange(1995, 2045)
    else:
        dates = df["ds"].copy(deep=True)
        years = list({x.year for x in dates})

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            holiday_names = getattr(hdays_part2, country)(years=years).values()
    except AttributeError:
        try:
            holiday_names = getattr(pyholidays, country)(years=years).values()
        except AttributeError:
            raise AttributeError(f"Holidays in {country} are not currently supported!")
    return set(holiday_names)


def events_config_to_model_dims(events_config, country_holidays_config):
    """
    Convert user specified events configurations along with country specific
        holidays to input dims for TimeNet model.

    Parameters
    ----------
        events_config : OrderedDict
            Configurations (upper, lower windows, regularization) for user specified events
        country_holidays_config : configure.Holidays
            Configurations (holiday_names, upper, lower windows, regularization) for country specific holidays

    Returns
    -------
        OrderedDict
            input dims for TimeNet model

            Note
            ----

            This dictionaries' keys correspond to individual holidays and contains configs such as the mode,
            list of event delims of the event corresponding to the offsets and
            indices in the input dataframe corresponding to each event.
    """
    if events_config is None and country_holidays_config is None:
        return None
    additive_events_dims = pd.DataFrame(columns=["event", "event_delim"])
    multiplicative_events_dims = pd.DataFrame(columns=["event", "event_delim"])

    if events_config is not None:
        for event, configs in events_config.items():
            mode = configs.mode
            for offset in range(configs.lower_window, configs.upper_window + 1):
                event_delim = create_event_names_for_offsets(event, offset)
                if mode == "additive":
                    additive_events_dims = pd.concat(
                        [
                            additive_events_dims,
                            pd.DataFrame([{"event": event, "event_delim": event_delim}]),
                        ],
                        ignore_index=True,
                    )
                else:
                    multiplicative_events_dims = pd.concat(
                        [multiplicative_events_dims, pd.DataFrame([{"event": event, "event_delim": event_delim}])],
                        ignore_index=True,
                    )

    if country_holidays_config is not None:
        lower_window = country_holidays_config.lower_window
        upper_window = country_holidays_config.upper_window
        mode = country_holidays_config.mode
        for country_holiday in country_holidays_config.holiday_names:
            for offset in range(lower_window, upper_window + 1):
                holiday_delim = create_event_names_for_offsets(country_holiday, offset)
                if mode == "additive":
                    additive_events_dims = pd.concat(
                        [
                            additive_events_dims,
                            pd.DataFrame([{"event": country_holiday, "event_delim": holiday_delim}]),
                        ],
                        ignore_index=True,
                    )
                else:
                    multiplicative_events_dims = pd.concat(
                        [
                            multiplicative_events_dims,
                            pd.DataFrame([{"event": country_holiday, "event_delim": holiday_delim}]),
                        ],
                        ignore_index=True,
                    )

    # sort based on event_delim
    event_dims = pd.DataFrame()
    if not additive_events_dims.empty:
        additive_events_dims = additive_events_dims.sort_values(by="event_delim").reset_index(drop=True)
        additive_events_dims["mode"] = "additive"
        event_dims = additive_events_dims

    if not multiplicative_events_dims.empty:
        multiplicative_events_dims = multiplicative_events_dims.sort_values(by="event_delim").reset_index(drop=True)
        multiplicative_events_dims["mode"] = "multiplicative"
        event_dims = pd.concat([event_dims, multiplicative_events_dims])

    event_dims_dic = OrderedDict({})
    # convert to dict format
    for event, row in event_dims.groupby("event"):
        event_dims_dic[event] = {
            "mode": row["mode"].iloc[0],
            "event_delim": list(row["event_delim"]),
            "event_indices": list(row.index),
        }
    return event_dims_dic


def create_event_names_for_offsets(event_name, offset):
    """
    Create names for offsets of every event

    Parameters
    ----------
        event_name : string
            Name of the event
        offset : int
            Offset of the event

    Returns
    -------
        string
            Name created for the offset of the event
    """
    offset_name = "{}_{}{}".format(event_name, "+" if offset >= 0 else "-", abs(offset))
    return offset_name


def regressors_config_to_model_dims(regressors_config):
    """
    Convert the NeuralProphet user specified regressors configurations to input dims for TimeNet model.

    Parameters
    ----------
        regressors_config : OrderedDict
            Configurations for user specified regressors

    Returns
    -------
        OrderedDict
            Input dims for TimeNet model.

            Note
            ----

            This dictionaries' keys correspond to individual regressor and values in a dict containing the mode
            and the indices in the input dataframe corresponding to each regressor.
    """
    if regressors_config is None:
        return None
    else:
        additive_regressors = []
        multiplicative_regressors = []

        if regressors_config is not None:
            for regressor, configs in regressors_config.items():
                mode = configs.mode
                if mode == "additive":
                    additive_regressors.append(regressor)
                else:
                    multiplicative_regressors.append(regressor)

        # sort based on event_delim
        regressors_dims = pd.DataFrame()
        if additive_regressors:
            additive_regressors = sorted(additive_regressors)
            additive_regressors_dims = pd.DataFrame(data=additive_regressors, columns=["regressors"])
            additive_regressors_dims["mode"] = "additive"
            regressors_dims = additive_regressors_dims

        if multiplicative_regressors:
            multiplicative_regressors = sorted(multiplicative_regressors)
            multiplicative_regressors_dims = pd.DataFrame(data=multiplicative_regressors, columns=["regressors"])
            multiplicative_regressors_dims["mode"] = "multiplicative"
            regressors_dims = pd.concat([regressors_dims, multiplicative_regressors_dims])

        regressors_dims_dic = OrderedDict({})
        # convert to dict format
        for index, row in regressors_dims.iterrows():
            regressors_dims_dic[row["regressors"]] = {"mode": row["mode"], "regressor_index": index}
        return regressors_dims_dic


def set_auto_seasonalities(df, season_config):
    """Set seasonalities that were left on auto or set by user.

    Note
    ----
    Turns on yearly seasonality if there is >=2 years of history.

    Turns on weekly seasonality if there is >=2 weeks of history, and the
    spacing between dates in the history is <7 days.

    Turns on daily seasonality if there is >=2 days of history, and the
    spacing between dates in the history is <1 day.

    Parameters
    ----------
        df : Dataframe
            Dataframe from which datestamps will be retrieved from
        season_config : configure.AllSeason
            NeuralProphet seasonal model configuration, as after __init__
    Returns
    -------
        configure.AllSeason
            Processed NeuralProphet seasonal model configuration

    """
    dates = df["ds"].copy(deep=True)

    log.debug(f"seasonality config received: {season_config}")
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
        if arg == "custom":
            continue
        elif arg == "auto":
            resolution = 0
            if auto_disable[name]:
                log.info(
                    f"Disabling {name} seasonality. Run NeuralProphet with "
                    f"{name}_seasonality=True to override this."
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
    season_config = season_config if len(season_config.periods) > 0 else None
    log.debug(f"seasonality config: {season_config}")
    return season_config


def print_epoch_metrics(metrics, val_metrics=None, e=0):
    if val_metrics is not None and len(val_metrics) > 0:
        val = OrderedDict({f"{key}_val": value for key, value in val_metrics.items()})
        metrics = {**metrics, **val}
    metrics_df = pd.DataFrame(
        {
            **metrics,
        },
        index=[e + 1],
    )
    metrics_string = metrics_df.to_string(float_format=lambda x: "{:6.3f}".format(x))
    return metrics_string


def fcst_df_to_last_forecast(fcst, n_last=1):
    """Converts from line-per-lag to line-per-forecast.

    Parameters
    ----------
        fcst : pd.DataFrame
            Forecast df
        n_last : int
            Number of last forecasts to include

    Returns
    -------
        pd.DataFrame
            Dataframe where yhat1 is last forecast, yhat2 second to last etc
    """

    cols = ["ds", "y"]  # cols to keep from df
    df = pd.concat((fcst[cols],), axis=1)
    df.reset_index(drop=True, inplace=True)

    yhat_col_names = [col_name for col_name in fcst.columns if "yhat" in col_name and "%" not in col_name]
    n_forecast_steps = len(yhat_col_names)
    yhats = pd.concat((fcst[yhat_col_names],), axis=1)
    cols = list(range(n_forecast_steps))
    for i in range(n_last - 1, -1, -1):
        forecast_name = f"yhat{i+1}"
        df[forecast_name] = None
        rows = len(df) + np.arange(-n_forecast_steps - i, -i, 1)
        last = yhats.values[rows, cols]
        df.loc[rows, forecast_name] = last
    return df


def set_y_as_percent(ax):
    """Set y axis as percentage

    Parameters
    ----------
        ax : matplotlib axis
            Respective y axis element

    Returns
    -------
        matplotlib axis
            Manipulated axis element
    """
    warnings.filterwarnings(
        action="ignore", category=UserWarning
    )  # workaround until there is clear direction how to handle this recent matplotlib bug
    yticks = 100 * ax.get_yticks()
    yticklabels = [f"{y:.4g}%" for y in yticks]
    ax.set_yticklabels(yticklabels)
    return ax


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def set_random_seed(seed=0):
    """Sets the random number generator to a fixed seed.

    Parameters
    ----------

    seed : numeric
        Seed value for random number generator

    Note
    ----
    This needs to be set each time before fitting the model.

    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_logger_level(logger, log_level, include_handlers=False):
    if log_level is None:
        logger.error("Failed to set log_level to None.")
    elif log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", 10, 20, 30, 40, 50):
        logger.error(
            "Failed to set log_level to {}."
            "Please specify a valid log level from: "
            "'DEBUG', 'INFO', 'WARNING', 'ERROR' or 'CRITICAL'"
            "".format(log_level)
        )
    else:
        logger.setLevel(log_level)
        if include_handlers:
            for h in log.handlers:
                h.setLevel(log_level)
        logger.debug(f"Set log level to {log_level}")


def set_log_level(log_level="INFO", include_handlers=False):
    """Set the log level of all logger objects

    Parameters
    ----------
        log_level : str
            The log level of the logger objects used for printing procedure status
            updates for debugging/monitoring. Should be one of ``NOTSET``, ``DEBUG``, ``INFO``, ``WARNING``,
            ``ERROR`` or ``CRITICAL``
        include_handlers : bool
            Include any specified file/stream handlers
    """
    set_logger_level(logging.getLogger("NP"), log_level, include_handlers)
