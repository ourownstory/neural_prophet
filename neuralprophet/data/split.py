import logging
from typing import Optional
from typing import OrderedDict as OrderedDictType
from typing import Tuple

import pandas as pd

from neuralprophet import df_utils
from neuralprophet.configure import ConfigEvents, Regressor
from neuralprophet.data.process import _check_dataframe

log = logging.getLogger("NP.data.splitting")


def _maybe_extend_df(
    df: pd.DataFrame,
    n_forecasts: int,
    max_lags: int,
    freq: Optional[str],
    config_regressors: Optional[OrderedDictType[str, Regressor]],
    config_events: Optional[ConfigEvents],
) -> Tuple[pd.DataFrame, dict]:
    """
    Extend the input DataFrame based on the number of forecasts, maximum lags,
    frequency, regressor configuration, and event configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to be extended.
    n_forecasts : int
        Number of forecasts to be made.
    max_lags : int
        Number of steps ahead of prediction time step to forecast.
    freq : str
        Frequency of the time series data.
    config_regressors : OrderedDict[str, Regressor]
        Configuration of regressors.
    config_events : ConfigEvents
        Configuration of events.

    Returns
    -------
    tuple[pd.DataFrame, int]
        A tuple containing the extended DataFrame and the periods added.
    """
    # Receives df with ID column
    periods_add = {}
    extended_df = pd.DataFrame()
    for df_name, df_i in df.groupby("ID"):
        _ = df_utils.infer_frequency(df_i, n_lags=max_lags, freq=freq)
        # to get all forecasteable values with df given, maybe extend into future:
        periods_add[df_name] = _get_maybe_extend_periods(
            df=df_i, n_forecasts=n_forecasts, max_lags=max_lags, config_regressors=config_regressors
        )
        if periods_add[df_name] > 0:
            # This does not include future regressors or events.
            # periods should be 0 if those are configured.
            last_date = pd.to_datetime(df_i["ds"].copy(deep=True)).sort_values().max()
            future_df = df_utils.make_future_df(
                df_columns=df_i.columns,
                last_date=last_date,
                periods=periods_add[df_name],
                freq=freq,
                config_events=config_events,
                config_regressors=config_regressors,
            )
            future_df["ID"] = df_name
            df_i = pd.concat([df_i, future_df])
            df_i.reset_index(drop=True, inplace=True)
        extended_df = pd.concat((extended_df, df_i.copy(deep=True)), ignore_index=True)
    return extended_df, periods_add


def _get_maybe_extend_periods(
    df: pd.DataFrame,
    n_forecasts: int,
    max_lags: int,
    config_regressors: Optional[OrderedDictType[str, Regressor]],
) -> int:
    """
    Determine the number of periods to extend the input DataFrame based on the
    number of forecasts, maximum lags, and regressor configuration.


    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a single ID column.
    n_forecasts : int
        Number of steps ahead of prediction time step to forecast.
    max_lags : int
        Maximum number of lags to consider.
    config_regressors : OrderedDictType[str, Regressor]
        Configuration of regressors. If None, the function may extend the
        DataFrame based on `n_forecasts` and `max_lags`.

    Returns
    -------
    int
        Number of periods to extend the input DataFrame.

    Raises
    ------
    AssertionError
        If the input DataFrame contains more than one unique ID.

    Notes
    -----
    The function assumes that the input DataFrame contains columns 'ID' and 'y'.
    """
    # Receives df with single ID column
    assert len(df["ID"].unique()) == 1
    periods_add = 0
    nan_at_end = 0
    while len(df) > nan_at_end and df["y"].isnull().iloc[-(1 + nan_at_end)]:
        nan_at_end += 1
    if max_lags > 0:
        if config_regressors is not None and config_regressors.regressors is not None:
            # if dataframe has already been extended into future,
            # don't extend beyond n_forecasts.
            periods_add = max(0, n_forecasts - nan_at_end)
        else:
            # can not extend as we lack future regressor values.
            periods_add = 0
    return periods_add


def _make_future_dataframe(
    model,
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    regressors_df: pd.DataFrame,
    periods: Optional[int],
    n_historic_predictions: int,
    n_forecasts: int,
    max_lags: int,
    freq: Optional[str],
) -> pd.DataFrame:
    """
    Generate a future dataframe by extending the input dataframe into the future.

    Parameters
    ----------
    model : NeuralProphet
        The model object used for prediction.
    df : pd.DataFrame
        The input dataframe with a single ID column and a 'ds' column containing timestamps.
    events_df : pd.DataFrame, optional
        The dataframe containing information about external events.
    regressors_df : pd.DataFrame, optional
        The dataframe containing information about external regressors.
    periods : int
        The number of steps to extend the DataFrame into the future.
    n_historic_predictions : int
        The number of historic predictions to include in the output dataframe.
    n_forecasts : int
        identical to NeuralProphet
    max_lags : int
        identical to NeuralProphet
    freq : str
        identical to NeuralProphet

    Returns
    -------
    pd.DataFrame
        The extended dataframe with additional rows for future periods.

    Raises
    ------
    ValueError
        If future values of all user specified regressors not provided.
    """
    # Receives df with single ID column
    assert len(df["ID"].unique()) == 1
    if periods == 0 and n_historic_predictions is True:
        log.warning(
            "Not extending df into future as no periods specified. You can skip this and predict directly instead."
        )
    df = df.copy(deep=True)
    _ = df_utils.infer_frequency(df, n_lags=max_lags, freq=freq)
    last_date = pd.to_datetime(df["ds"].copy(deep=True).dropna()).sort_values().max()
    if events_df is not None:
        events_df = events_df.copy(deep=True).reset_index(drop=True)
    if regressors_df is not None:
        regressors_df = regressors_df.copy(deep=True).reset_index(drop=True)
    if periods is None:
        periods = 1 if max_lags == 0 else n_forecasts
    else:
        assert periods >= 0

    if isinstance(n_historic_predictions, bool):
        if n_historic_predictions:
            n_historic_predictions = len(df) - max_lags
        else:
            n_historic_predictions = 0
    elif not isinstance(n_historic_predictions, int):
        log.error("non-integer value for n_historic_predictions set to zero.")
        n_historic_predictions = 0

    if periods == 0 and n_historic_predictions == 0:
        raise ValueError("Set either history or future to contain more than zero values.")

    # check for external regressors known in future
    if model.config_regressors.regressors is not None and periods > 0:
        if regressors_df is None:
            raise ValueError("Future values of all user specified regressors not provided")
        else:
            for regressor in model.config_regressors.regressors.keys():
                if regressor not in regressors_df.columns:
                    raise ValueError(f"Future values of user specified regressor {regressor} not provided")

    if len(df) < max_lags:
        raise ValueError(
            "Insufficient input data for a prediction."
            "Please supply historic observations (number of rows) of at least max_lags (max of number of n_lags)."
        )
    elif len(df) < max_lags + n_historic_predictions:
        log.warning(
            f"Insufficient data for {n_historic_predictions} historic forecasts, reduced to {len(df) - max_lags}."
        )
        n_historic_predictions = len(df) - max_lags
    if (n_historic_predictions + max_lags) == 0:
        df = pd.DataFrame(columns=df.columns)
    else:
        df = df[-(max_lags + n_historic_predictions) :]
        nan_at_end = 0
        while len(df) > nan_at_end and df["y"].isnull().iloc[-(1 + nan_at_end)]:
            nan_at_end += 1
        if nan_at_end > 0:
            if max_lags > 0 and (nan_at_end + 1) >= max_lags:
                raise ValueError(
                    f"{nan_at_end + 1} missing values were detected at the end of df before df was extended into "
                    "the future. Please make sure there are no NaN values at the end of df."
                )
            df["y"].iloc[-(nan_at_end + 1) :].ffill(inplace=True)
            log.warning(
                f"{nan_at_end + 1} missing values were forward-filled at the end of df before df was extended into the "
                "future. Please make sure there are no NaN values at the end of df."
            )

    if len(df) > 0:
        if len(df.columns) == 1 and "ds" in df:
            assert max_lags == 0
            df = _check_dataframe(model, df, check_y=False, exogenous=False)
        else:
            df = _check_dataframe(model, df, check_y=max_lags > 0, exogenous=True, future=True)
    # future data
    # check for external events known in future
    if model.config_events is not None and periods > 0 and events_df is None:
        log.warning(
            "Future values not supplied for user specified events. "
            "All events being treated as not occurring in future"
        )

    if max_lags > 0:
        if periods > 0 and periods != n_forecasts:
            periods = n_forecasts
            log.warning(f"Number of forecast steps is defined by n_forecasts. Adjusted to {n_forecasts}.")

    if periods > 0:
        future_df = df_utils.make_future_df(
            df_columns=df.columns,
            last_date=last_date,
            periods=periods,
            freq=freq,
            config_events=model.config_events,
            events_df=events_df,
            config_regressors=model.config_regressors,
            regressors_df=regressors_df,
        )
        if len(df) > 0:
            df = pd.concat([df, future_df])
        else:
            df = future_df
    df = df.reset_index(drop=True)
    model.predict_steps = periods
    return df
