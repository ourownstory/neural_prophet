import numpy as np
import pandas as pd
import torch


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


def parse_seasonality_args(seasonalities, name, arg, auto_disable, default_order):
    """Get number of fourier components for built-in seasonalities.

    Parameters
    ----------
    name: string name of the seasonality component.
    arg: 'auto', True, False, or number of fourier components as provided.
    auto_disable: bool if seasonality should be disabled when 'auto'.
    default_order: int default fourier order

    Returns
    -------
    Number of fourier components, or 0 for disabled.
    """
    if arg == 'auto':
        fourier_order = 0
        if name in seasonalities:
            pass
            # logger.info(
            #     'Found custom seasonality named {name!r}, disabling '
            #     'built-in {name!r} seasonality.'.format(name=name)
            # )
        elif auto_disable:
            pass
            # logger.info(
            #     'Disabling {name} seasonality. Run prophet with '
            #     '{name}_seasonality=True to override this.'
            #     .format(name=name)
            # )
        else:
            fourier_order = default_order
    elif arg is True:
        fourier_order = default_order
    elif arg is False:
        fourier_order = 0
    else:
        fourier_order = int(arg)
    return fourier_order


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
    return np.column_stack([
        fun((2.0 * (i + 1) * np.pi * t / period))
        for i in range(series_order)
        for fun in (np.sin, np.cos)
    ])


def make_seasonality_features(dates, period, series_order, prefix):
    """Data frame with seasonality features.

    Parameters
    ----------
    cls: Prophet class.
    dates: pd.Series containing timestamps.
    period: Number of days of the period.
    series_order: Number of components.
    prefix: Column name prefix.

    Returns
    -------
    pd.DataFrame with seasonality features.
    """
    features = fourier_series(dates, period, series_order)
    columns = [
        '{}_delim_{}'.format(prefix, i + 1)
        for i in range(features.shape[1])
    ]
    return pd.DataFrame(features, columns=columns)



def make_all_seasonality_features(df, seasonalities):
    """Dataframe with seasonality features.

    Includes seasonality features, holiday features, and added regressors.

    Parameters
    ----------
    df: pd.DataFrame with dates for computing seasonality features and any
        added regressors.

    Returns
    -------
    pd.DataFrame with regression features.
    list of prior scales for each column of the features dataframe.
    Dataframe with indicators for which regression components correspond to
        which columns.
    Dictionary with keys 'additive' and 'multiplicative' listing the
        component names for each mode of seasonality.
    """
    seasonal_features = []
    # prior_scales = []
    modes = {'additive': [], 'multiplicative': []}

    # Seasonality features
    for name, props in seasonalities.items():
        features = make_seasonality_features(
            df['ds'],
            props['period'],
            props['fourier_order'],
            name,
        )
        ## Future TODO: allow conditions for seasonality
        # if props['condition_name'] is not None:
        #     features[~df[props['condition_name']]] = 0
        seasonal_features.append(features)
        # prior_scales.extend(
        #     [props['prior_scale']] * features.shape[1])
        modes[props['mode']].append(name)

    # # Holiday features
    # holidays = self.construct_holiday_dataframe(df['ds'])
    # if len(holidays) > 0:
    #     features, holiday_priors, holiday_names = (
    #         self.make_holiday_features(df['ds'], holidays)
    #     )
    #     seasonal_features.append(features)
    #     prior_scales.extend(holiday_priors)
    #     modes[self.seasonality_mode].extend(holiday_names)

    # # Additional regressors
    # for name, props in self.extra_regressors.items():
    #     seasonal_features.append(pd.DataFrame(df[name]))
    #     prior_scales.append(props['prior_scale'])
    #     modes[props['mode']].append(name)

    # Dummy to prevent empty X
    if len(seasonal_features) == 0:
        seasonal_features.append(
            pd.DataFrame({'zeros': np.zeros(df.shape[0])}))
        # prior_scales.append(1.)

    seasonal_features = pd.concat(seasonal_features, axis=1)
    # component_cols, modes = self.regressor_column_matrix(
    #     seasonal_features, modes
    # )
    # return seasonal_features , prior_scales, component_cols, modes
    return seasonal_features, modes

