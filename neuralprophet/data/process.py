import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from neuralprophet import df_utils, time_dataset
from neuralprophet.configure import (
    ConfigCountryHolidays,
    ConfigEvents,
    ConfigFutureRegressors,
    ConfigLaggedRegressors,
    ConfigSeasonality,
)
from neuralprophet.np_types import Components

log = logging.getLogger("NP.data.processing")


def _reshape_raw_predictions_to_forecst_df(
    df: pd.DataFrame,
    predicted: np.ndarray,
    components: Optional[Components],
    prediction_frequency: Optional[dict],
    dates: pd.Series,
    n_forecasts: int,
    max_lags: int,
    freq: Optional[str],
    quantiles: List[float],
    config_lagged_regressors: Optional[ConfigLaggedRegressors],
) -> pd.DataFrame:
    """
    Turns forecast-origin-wise predictions into forecast-target-wise predictions.

    Parameters
    ----------
        df : pd.DataFrame
            input dataframe
        predicted : np.array
            Array containing the forecasts
        components : dict[np.array]
            Dictionary of components containing an array of each components' contribution to the forecast
        prediction_frequency : str
            Frequency of the predictions
        dates : pd.Series
            timestamps referring to the start of the predictions
        n_forecasts : int
            Number of steps ahead of prediction time step to forecast.
        max_lags : int
            Maximum number of lags to use
        freq : str
            Data step sizes. Frequency of data recording.
        quantiles : list[float]
            List of quantiles to include in the forecast
        config_lagged_regressors : ConfigLaggedRegressors
            Configuration for lagged regressors

    Returns
    -------
        pd.DataFrame
            columns ``ds``, ``y``, ``trend`` and [``yhat<i>``]

            Note
            ----
            where yhat<i> refers to the i-step-ahead prediction for this row's datetime.
            e.g. yhat3 is the prediction for this datetime, predicted 3 steps ago, "3 steps old".
    """
    # Receives df with single ID column
    assert len(df["ID"].unique()) == 1
    cols = ["ds", "y", "ID"]  # cols to keep from df
    df_forecast = pd.concat((df[cols],), axis=1)
    # create a line for each forecast_lag
    # 'yhat<i>' is the forecast for 'y' at 'ds' from i steps ago.
    for j in range(len(quantiles)):
        for forecast_lag in range(1, n_forecasts + 1):
            forecast = predicted[:, forecast_lag - 1, j]
            pad_before = max_lags + forecast_lag - 1
            pad_after = n_forecasts - forecast_lag
            yhat = np.pad(forecast, (pad_before, pad_after), mode="constant", constant_values=np.NaN)
            if prediction_frequency is not None:
                ds = df_forecast["ds"].iloc[pad_before : -pad_after if pad_after > 0 else None]
                mask = df_utils.create_mask_for_prediction_frequency(
                    prediction_frequency=prediction_frequency,
                    ds=ds,
                    forecast_lag=forecast_lag,
                )
                yhat = np.full((len(ds),), np.nan)
                yhat[mask] = forecast
                yhat = np.pad(yhat, (pad_before, pad_after), mode="constant", constant_values=np.NaN)
            # 0 is the median quantile index
            if j == 0:
                name = f"yhat{forecast_lag}"
            else:
                name = f"yhat{forecast_lag} {round(quantiles[j] * 100, 1)}%"
            df_forecast[name] = yhat

    if components is None:
        return df_forecast

    # else add components
    lagged_components = [
        "ar",
    ]
    if config_lagged_regressors is not None:
        for name in config_lagged_regressors.keys():
            lagged_components.append(f"lagged_regressor_{name}")
    for comp in lagged_components:
        if comp in components:
            for j in range(len(quantiles)):
                for forecast_lag in range(1, n_forecasts + 1):
                    forecast = components[comp][:, forecast_lag - 1, j]  # 0 is the median quantile
                    pad_before = max_lags + forecast_lag - 1
                    pad_after = n_forecasts - forecast_lag
                    yhat = np.pad(forecast, (pad_before, pad_after), mode="constant", constant_values=np.NaN)
                    if prediction_frequency is not None:
                        ds = df_forecast["ds"].iloc[pad_before : -pad_after if pad_after > 0 else None]
                        mask = df_utils.create_mask_for_prediction_frequency(
                            prediction_frequency=prediction_frequency,
                            ds=ds,
                            forecast_lag=forecast_lag,
                        )
                        yhat = np.full((len(ds),), np.nan)
                        yhat[mask] = forecast
                        yhat = np.pad(yhat, (pad_before, pad_after), mode="constant", constant_values=np.NaN)
                    if j == 0:  # temporary condition to add only the median component
                        name = f"{comp}{forecast_lag}"
                        df_forecast[name] = yhat

    # only for non-lagged components
    for comp in components:
        if comp not in lagged_components:
            for j in range(len(quantiles)):
                forecast_0 = components[comp][0, :, j]
                forecast_rest = components[comp][1:, n_forecasts - 1, j]
                yhat = np.pad(
                    np.concatenate((forecast_0, forecast_rest)), (max_lags, 0), mode="constant", constant_values=np.NaN
                )
                if prediction_frequency is not None:
                    date_list = []
                    for key, value in prediction_frequency.items():
                        if key == "daily-hour":
                            dates_comp = dates[dates.dt.hour == value]
                        elif key == "weekly-day":
                            dates_comp = dates[dates.dt.dayofweek == value]
                        elif key == "monthly-day":
                            dates_comp = dates[dates.dt.day == value]
                        elif key == "yearly-month":
                            dates_comp = dates[dates.dt.month == value]
                        elif key == "hourly-minute":
                            dates_comp = dates[dates.dt.minute == value]
                        else:
                            raise ValueError(f"prediction_frequency {key} not supported")
                        date_list.append(dates_comp)
                    # create new pd.Series only containing the dates that are in all Series in date_list
                    dates_comp = pd.Series(date_list[0])
                    for i in range(1, len(date_list)):
                        dates_comp = dates_comp[dates_comp.isin(date_list[i])]
                    ser = pd.Series(dtype="datetime64[ns]")
                    for date in dates_comp:
                        d = pd.date_range(date, periods=n_forecasts + 1, freq=freq)
                        ser = pd.concat((ser, pd.Series(d).iloc[1:]))
                    df_comp = pd.DataFrame({"ds": ser, "yhat": components[comp][:, :, j].flatten()}).drop_duplicates(
                        subset="ds"
                    )
                    df_comp, _ = df_utils.add_missing_dates_nan(df=df_comp, freq=freq)
                    yhat = pd.merge(df_forecast.filter(["ds", "ID"]), df_comp, on="ds", how="left")["yhat"].values
                if j == 0:  # temporary condition to add only the median component
                    # add yhat into dataframe, using df_forecast indexing
                    yhat_df = pd.Series(yhat, name=comp).set_axis(df_forecast.index)
                    df_forecast = pd.concat([df_forecast, yhat_df], axis=1, ignore_index=False)
    return df_forecast


def _convert_raw_predictions_to_raw_df(
    dates: pd.Series,
    predicted: np.ndarray,
    n_forecasts: int,
    quantiles: List[float],
    components: Optional[Components] = None,
) -> pd.DataFrame:
    """Turns forecast-origin-wise predictions into forecast-target-wise predictions.

    Parameters
    ----------
        dates : pd.Series
            timestamps referring to the start of the predictions.
        predicted : np.array
            Array containing the forecasts
        n_forecasts : int
            optional, number of steps ahead of prediction time step to forecast
        quantiles : list[float]
            optional, list of quantiles for quantile regression uncertainty estimate
        components : dict[np.array]
            Dictionary of components containing an array of each components' contribution to the forecast

    Returns
    -------
        pd. DataFrame
            columns ``ds``, ``y``, and [``step<i>``]

            Note
            ----
            where step<i> refers to the i-step-ahead prediction *made at* this row's datetime.
            e.g. the first forecast step0 is the prediction for this timestamp,
            the step1 is for the timestamp after, ...
            ... step3 is the prediction for 3 steps into the future,
            predicted using information up to (excluding) this datetime.
    """
    all_data = predicted
    df_raw = pd.DataFrame()
    df_raw.insert(0, "ds", dates.values)
    df_raw.insert(1, "ID", "__df__")  # type: ignore
    for forecast_lag in range(n_forecasts):
        for quantile_idx in range(len(quantiles)):
            # 0 is the median quantile index
            if quantile_idx == 0:
                step_name = f"step{forecast_lag}"
            else:
                step_name = f"step{forecast_lag} {quantiles[quantile_idx] * 100}%"
            data = all_data[:, forecast_lag, quantile_idx]
            ser = pd.Series(data=data, name=step_name)
            df_raw = df_raw.merge(ser, left_index=True, right_index=True)
        if components is not None:
            for comp_name, comp_data in components.items():
                comp_name_ = f"{comp_name}{forecast_lag}"
                data = comp_data[:, forecast_lag, 0]  # for components the quantiles are ignored for now
                ser = pd.Series(data=data, name=comp_name_)
                df_raw = df_raw.merge(ser, left_index=True, right_index=True)
    return df_raw


def _prepare_dataframe_to_predict(model, df: pd.DataFrame, max_lags: int, freq: Optional[str]) -> pd.DataFrame:
    """
    Pre-processes a dataframe for prediction using the specified model.

    Parameters
    ----------
        model:
            The NeuralProphet model
        df: pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
        max_lags: int
            The maximum number of lags to include in the output dataframe.
        freq: str
            data step sizes. Frequency of data recording,

    Returns
    ----------
        pd.DataFrame
            pre-processed dataframe

    Raises
    ----------
        ValueError
        If the input dataframe has already been normalized, if there is insufficient input data for prediction,
        if only datestamps are provided but y values are needed for auto-regression.
    """
    # Receives df with ID column
    df_prepared = pd.DataFrame()
    for df_name, df_i in df.groupby("ID"):
        df_i = df_i.copy(deep=True)
        _ = df_utils.infer_frequency(df_i, n_lags=max_lags, freq=freq)
        # check if received pre-processed df
        if "y_scaled" in df_i.columns or "t" in df_i.columns:
            raise ValueError(
                "DataFrame has already been normalized. " "Please provide raw dataframe or future dataframe."
            )
        # Checks
        if len(df_i) == 0 or len(df_i) < max_lags:
            raise ValueError(
                "Insufficient input data for a prediction."
                "Please supply historic observations (number of rows) of at least max_lags (max of number of n_lags)."
            )
        if len(df_i.columns) == 1 and "ds" in df_i:
            if max_lags != 0:
                raise ValueError("only datestamps provided but y values needed for auto-regression.")
            df_i = _check_dataframe(model, df_i, check_y=False, exogenous=False)
        else:
            df_i = _check_dataframe(model, df_i, check_y=model.max_lags > 0, exogenous=False)
            # fill in missing nans except for nans at end
            df_i = _handle_missing_data(
                df=df_i,
                freq=freq,
                n_lags=model.n_lags,
                n_forecasts=model.n_forecasts,
                config_missing=model.config_missing,
                config_regressors=model.config_regressors,
                config_lagged_regressors=model.config_lagged_regressors,
                config_events=model.config_events,
                config_seasonality=model.config_seasonality,
                predicting=True,
            )
        df_prepared = pd.concat((df_prepared, df_i.copy(deep=True).reset_index(drop=True)), ignore_index=True)
    return df_prepared


def _validate_column_name(
    name: str,
    config_events: Optional[ConfigEvents],
    config_country_holidays: Optional[ConfigCountryHolidays],
    config_seasonality: Optional[ConfigSeasonality],
    config_lagged_regressors: Optional[ConfigLaggedRegressors],
    config_regressors: Optional[ConfigFutureRegressors],
    events: Optional[bool] = True,
    seasons: Optional[bool] = True,
    regressors: Optional[bool] = True,
    covariates: Optional[bool] = True,
):
    """Validates the name of a seasonality, event, or regressor.

    Parameters
    ----------
        name : str
            name of seasonality, event or regressor
        config_events : Optional[ConfigEvents]
            Configuration options for adding events to the model.
        config_country_holidays : Optional[ConfigCountryHolidays]
            Configuration options for adding country holidays to the model.
        config_seasonality : Optional[ConfigSeasonality]
            Configuration options for adding seasonal components to the model.
        config_lagged_regressors : Optional[ConfigLaggedRegressors]
            Configuration options for adding lagged external regressors to the model.
        config_regressors : Optional[ConfigFutureRegressors]
            Configuration options for adding future regressors to the model.
        events : bool
            check if name already used for event
        seasons : bool
            check if name already used for seasonality
        regressors : bool
            check if name already used for regressor
        covariates : bool
            check if name already used for covariate
    """
    reserved_names = [
        "trend",
        "daily",
        "weekly",
        "yearly",
        "events",
        "holidays",
        "yhat",
        "ID",
        "y_scaled",
        "ds",
        "t",
        "y",
        "index",
    ]
    rn_l = [n + "_lower" for n in reserved_names]
    rn_u = [n + "_upper" for n in reserved_names]
    reserved_names.extend(rn_l)
    reserved_names.extend(rn_u)
    reserved_names.extend(["ds", "y", "cap", "floor", "y_scaled", "cap_scaled"])
    if name in reserved_names:
        raise ValueError(f"Name {name!r} is reserved.")
    if events and config_events is not None:
        if name in config_events.keys():
            raise ValueError(f"Name {name!r} already used for an event.")
    if events and config_country_holidays is not None:
        if name in config_country_holidays.holiday_names:
            raise ValueError(f"Name {name!r} is a holiday name in {config_country_holidays.country}.")
    if seasons and config_seasonality is not None:
        if name in config_seasonality.periods:
            raise ValueError(f"Name {name!r} already used for a seasonality.")
    if covariates and config_lagged_regressors is not None:
        if name in config_lagged_regressors:
            raise ValueError(f"Name {name!r} already used for an added covariate.")
    if regressors and config_regressors.regressors is not None:
        if name in config_regressors.regressors.keys():
            raise ValueError(f"Name {name!r} already used for an added regressor.")


def _check_dataframe(
    model,
    df: pd.DataFrame,
    check_y: bool = True,
    exogenous: bool = True,
    future: Optional[bool] = None,
) -> pd.DataFrame:
    """Performs basic data sanity checks and ordering

    Prepare dataframe for fitting or predicting.

    Parameters
    ----------
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data
        check_y : bool
            if df must have series values

            Note
            ----
            set to True if training or predicting with autoregression
        exogenous : bool
            whether to check covariates, regressors and events column names
        future : bool
            whether this function is called by make_future_dataframe()

    Returns
    -------
        pd.DataFrame
            checked dataframe
    """
    if len(df) < (model.n_forecasts + model.n_lags) and not future:
        raise ValueError(
            "Dataframe has less than n_forecasts + n_lags rows. "
            "Forecasting not possible. Please either use a larger dataset, or adjust the model parameters."
        )
    df, _, _, _ = df_utils.prep_or_copy_df(df)
    df, regressors_to_remove, lag_regressors_to_remove = df_utils.check_dataframe(
        df=df,
        check_y=check_y,
        covariates=model.config_lagged_regressors if exogenous else None,
        regressors=model.config_regressors.regressors if exogenous else None,
        events=model.config_events if exogenous else None,
        seasonalities=model.config_seasonality if exogenous else None,
        future=True if future else None,
    )

    if model.config_regressors.regressors is not None:
        for reg in regressors_to_remove:
            log.warning(f"Removing regressor {reg} because it is not present in the data.")
            model.config_regressors.regressors.pop(reg)
        if model.config_regressors.regressors is not None and len(model.config_regressors.regressors) == 0:
            model.config_regressors.regressors = None
    if model.config_lagged_regressors is not None:
        for reg in lag_regressors_to_remove:
            log.warning(f"Removing lagged regressor {reg} because it is not present in the data.")
            model.config_lagged_regressors.pop(reg)
        if len(model.config_lagged_regressors) == 0:
            model.config_lagged_regressors = None
    return df


def _handle_missing_data(
    df: pd.DataFrame,
    freq: str,
    n_lags: int,
    n_forecasts: int,
    config_missing,
    config_regressors: Optional[ConfigFutureRegressors] = None,
    config_lagged_regressors: Optional[ConfigLaggedRegressors] = None,
    config_events: Optional[ConfigEvents] = None,
    config_seasonality: Optional[ConfigSeasonality] = None,
    predicting: bool = False,
) -> pd.DataFrame:
    """
    Checks and normalizes new data, auto-imputing missing data if `config_missing` allows it.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing columns 'ds', 'y', and optionally 'ID' with all data.
    freq : str
            data step sizes. Frequency of data recording,

            Note
            ----
            Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto`` (default) to
            automatically set frequency.
    n_lags : int
        Previous time series steps to include in auto-regression. Aka AR-order
    n_forecasts : int
        Number of steps ahead of prediction time step to forecast.
    config_missing :
        Configuration options for handling missing data.
    config_regressors : Optional[ConfigFutureRegressors]
        Configuration options for adding future regressors to the model.
    config_lagged_regressors : Optional[ConfigLaggedRegressors]
        Configuration options for adding lagged external regressors to the model.
    config_events : Optional[ConfigEvents]
        Configuration options for adding events to the model.
    config_seasonality : Optional[ConfigSeasonality]
        Configuration options for adding seasonal components to the model.
    predicting : bool, default False
        If True, allows missing values in the 'y' column for the forecast period, or missing completely.

    Returns
    -------
    pd.DataFrame
        The pre-processed DataFrame, including imputed missing data, if applicable.

    """
    df, _, _, _ = df_utils.prep_or_copy_df(df)

    if n_lags == 0 and not predicting:
        # drop rows with NaNs in y and count them
        df_na_dropped = df.dropna(subset=["y"])
        n_dropped = len(df) - len(df_na_dropped)
        if n_dropped > 0:
            df = df_na_dropped
            log.info(f"Dropped {n_dropped} rows with NaNs in 'y' column.")

    if n_lags > 0:
        # add missig dates to df
        df_grouped = df.groupby("ID").apply(lambda x: x.set_index("ds").resample(freq).asfreq()).drop(columns=["ID"])
        n_missing_dates = len(df_grouped) - len(df)
        if n_missing_dates > 0:
            df = df_grouped.reset_index()
            log.info(f"Added {n_missing_dates} missing dates.")

    if config_regressors is not None and config_regressors.regressors is not None:
        # drop complete row for future regressors that are NaN at the end
        last_valid_index = df.groupby("ID")[list(config_regressors.regressors.keys())].apply(
            lambda x: x.last_valid_index()
        )
        df_dropped = df.groupby("ID", group_keys=False).apply(lambda x: x.loc[: last_valid_index[x.name]])
        n_dropped = len(df) - len(df_dropped)
        if n_dropped > 0:
            df = df_dropped
            log.info(f"Dropped {n_dropped} rows at the end with NaNs in future regressors.")

    dropped_trailing_y = False
    if df["y"].isna().any():
        # drop complete row if y of ID ends with nan
        last_valid_index = df.groupby("ID")["y"].apply(lambda x: x.last_valid_index())
        df_dropped = df.groupby("ID", group_keys=False).apply(lambda x: x.loc[: last_valid_index[x.name]])
        n_dropped = len(df) - len(df_dropped)
        if n_dropped > 0:
            dropped_trailing_y = True
            # save dropped rows for later
            df_to_add = df.groupby("ID", group_keys=False).apply(lambda x: x.loc[last_valid_index[x.name] + 1 :])
            df = df_dropped
            log.info(f"Dropped {n_dropped} rows at the end with NaNs in 'y' column.")

    if config_missing.impute_missing:
        # impute missing values
        data_columns = []
        if n_lags > 0:
            data_columns.append("y")
        if config_lagged_regressors is not None:
            data_columns.extend(config_lagged_regressors.keys())
        if config_regressors is not None and config_regressors.regressors is not None:
            data_columns.extend(config_regressors.regressors.keys())
        if config_events is not None:
            data_columns.extend(config_events.keys())
        conditional_cols = []
        if config_seasonality is not None:
            conditional_cols = list(
                set(
                    [
                        value.condition_name
                        for key, value in config_seasonality.periods.items()
                        if value.condition_name is not None
                    ]
                )
            )
            data_columns.extend(conditional_cols)
        for column in data_columns:
            sum_na = df[column].isna().sum()
            if sum_na > 0:
                log.warning(f"{sum_na} missing values in column {column} were detected in total. ")
                # use 0 substitution for holidays and events missing values
                if config_events is not None and column in config_events.keys():
                    df[column].fillna(0, inplace=True)
                    remaining_na = 0
                else:
                    df.loc[:, column], remaining_na = df_utils.fill_linear_then_rolling_avg(
                        df[column],
                        limit_linear=config_missing.impute_linear,
                        rolling=config_missing.impute_rolling,
                    )
                log.info(f"{sum_na - remaining_na} NaN values in column {column} were auto-imputed.")
                if remaining_na > 0:
                    log.warning(
                        f"More than {2 * config_missing.impute_linear + config_missing.impute_rolling} consecutive \
                            missing values encountered in column {column}. "
                        f"{remaining_na} NA remain after auto-imputation. "
                    )
    if dropped_trailing_y and predicting:
        # add trailing y values again if in predict mode
        df = pd.concat([df, df_to_add])
        if config_seasonality is not None and len(conditional_cols) > 0:
            df[conditional_cols] = df[conditional_cols].ffill()  # type: ignore
    return df


def _create_dataset(model, df, predict_mode, prediction_frequency=None):
    """Construct dataset from dataframe.

    (Configured Hyperparameters can be overridden by explicitly supplying them.
    Useful to predict a single model component.)

    Parameters
    ----------
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally``ID`` and
            normalized columns normalized columns ``ds``, ``y``, ``t``, ``y_scaled``
        predict_mode : bool
            specifies predict mode

            Options
                * ``False``: includes target values.
                * ``True``: does not include targets but includes entire dataset as input

        prediction_frequency: dict
            periodic interval in which forecasts should be made.
            Key: str
                periodicity of the predictions to be made, e.g. 'daily-hour'.

            Options
                * ``'hourly-minute'``: forecast once per hour at a specified minute
                * ``'daily-hour'``: forecast once per day at a specified hour
                * ``'weekly-day'``: forecast once per week at a specified day
                * ``'monthly-day'``: forecast once per month at a specified day
                * ``'yearly-month'``: forecast once per year at a specified month

            value: int
                forecast origin of the predictions to be made, e.g. 7 for 7am in case of 'daily-hour'.

    Returns
    -------
        TimeDataset
    """
    df, _, _, _ = df_utils.prep_or_copy_df(df)
    return time_dataset.GlobalTimeDataset(
        df,
        predict_mode=predict_mode,
        n_lags=model.n_lags,
        n_forecasts=model.n_forecasts,
        prediction_frequency=prediction_frequency,
        predict_steps=model.predict_steps,
        config_seasonality=model.config_seasonality,
        config_events=model.config_events,
        config_country_holidays=model.config_country_holidays,
        config_regressors=model.config_regressors,
        config_lagged_regressors=model.config_lagged_regressors,
        config_missing=model.config_missing,
        # config_train=model.config_train, # no longer needed since JIT tabularization.
    )
