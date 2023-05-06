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
            yhat = np.concatenate(([np.NaN] * pad_before, forecast, [np.NaN] * pad_after))
            if prediction_frequency is not None:
                ds = df_forecast["ds"].iloc[pad_before : -pad_after if pad_after > 0 else None]
                mask = df_utils.create_mask_for_prediction_frequency(
                    prediction_frequency=prediction_frequency,
                    ds=ds,
                    forecast_lag=forecast_lag,
                )
                yhat = np.full((len(ds),), np.nan)
                yhat[mask] = forecast
                yhat = np.concatenate(([np.NaN] * pad_before, yhat, [np.NaN] * pad_after))
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
                    yhat = np.concatenate(([np.NaN] * pad_before, forecast, [np.NaN] * pad_after))
                    if prediction_frequency is not None:
                        ds = df_forecast["ds"].iloc[pad_before : -pad_after if pad_after > 0 else None]
                        mask = df_utils.create_mask_for_prediction_frequency(
                            prediction_frequency=prediction_frequency,
                            ds=ds,
                            forecast_lag=forecast_lag,
                        )
                        yhat = np.full((len(ds),), np.nan)
                        yhat[mask] = forecast
                        yhat = np.concatenate(([np.NaN] * pad_before, yhat, [np.NaN] * pad_after))
                    if j == 0:  # temporary condition to add only the median component
                        name = f"{comp}{forecast_lag}"
                        df_forecast[name] = yhat

    # only for non-lagged components
    for comp in components:
        if comp not in lagged_components:
            for j in range(len(quantiles)):
                forecast_0 = components[comp][0, :, j]
                forecast_rest = components[comp][1:, n_forecasts - 1, j]
                yhat = np.concatenate(([np.NaN] * max_lags, forecast_0, forecast_rest))
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
                    df_comp = pd.DataFrame({"ds": ser, "yhat": components[comp].flatten()}).drop_duplicates(subset="ds")
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
            df_i, regressors_to_remove, lag_regressors_to_remove, dummy_ds_activated = df_utils.check_dataframe(
                df=df_i,
                n_forecasts=model.n_forecasts,
                n_lags=model.n_lags,
                check_y=False,
                covariates=None,
                regressors=None,
                events=None,
                seasonalities=None,
            )
            # Adjusting model properties
            if model.config_seasonality is not None and dummy_ds_activated is True:
                for name, period in model.config_seasonality.periods.items():
                    resolution = 0
                    log.warning(f"Disabling {name} seasonality due to missing datestamps.")
                    model.config_seasonality.periods[name].resolution = resolution
            if model.config_regressors is not None:
                for reg in regressors_to_remove:
                    log.warning(f"Removing regressor {reg} because it is not present in the data.")
                    model.config_regressors.pop(reg)
                if len(model.config_regressors) == 0:
                    model.config_regressors = None
            if model.config_lagged_regressors is not None:
                for reg in lag_regressors_to_remove:
                    log.warning(f"Removing lagged regressor {reg} because it is not present in the data.")
                    model.config_lagged_regressors.pop(reg)
                if len(model.config_lagged_regressors) == 0:
                    model.config_lagged_regressors = None
        else:
            df_i, regressors_to_remove, lag_regressors_to_remove, dummy_ds_activated = df_utils.check_dataframe(
                df=df_i,
                n_forecasts=model.n_forecasts,
                n_lags=model.n_lags,
                check_y=model.max_lags > 0,
                covariates=None,
                regressors=None,
                events=None,
                seasonalities=None,
            )
            # Adjusting model properties
            if model.config_seasonality is not None and dummy_ds_activated is True:
                for name, period in model.config_seasonality.periods.items():
                    resolution = 0
                    log.warning(f"Disabling {name} seasonality due to missing datestamps.")
                    model.config_seasonality.periods[name].resolution = resolution
            if model.config_regressors is not None:
                for reg in regressors_to_remove:
                    log.warning(f"Removing regressor {reg} because it is not present in the data.")
                    model.config_regressors.pop(reg)
                if len(model.config_regressors) == 0:
                    model.config_regressors = None
            if model.config_lagged_regressors is not None:
                for reg in lag_regressors_to_remove:
                    log.warning(f"Removing lagged regressor {reg} because it is not present in the data.")
                    model.config_lagged_regressors.pop(reg)
                if len(model.config_lagged_regressors) == 0:
                    model.config_lagged_regressors = None
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
        "additive_terms",
        "daily",
        "weekly",
        "yearly",
        "events",
        "holidays",
        "zeros",
        "extra_regressors_additive",
        "yhat",
        "extra_regressors_multiplicative",
        "multiplicative_terms",
        "ID",
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
    if regressors and config_regressors is not None:
        if name in config_regressors.keys():
            raise ValueError(f"Name {name!r} already used for an added regressor.")


def _handle_missing_data(
    df: pd.DataFrame,
    freq: Optional[str],
    n_lags: int,
    n_forecasts: int,
    config_missing,
    config_regressors: Optional[ConfigFutureRegressors],
    config_lagged_regressors: Optional[ConfigLaggedRegressors],
    config_events: Optional[ConfigEvents],
    config_seasonality: Optional[ConfigSeasonality],
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
            Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto`` (default) to automatically set frequency.
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
    df_handled_missing = pd.DataFrame()
    for df_name, df_i in df.groupby("ID"):
        df_handled_missing_aux = _handle_missing_data_single_id(
            df=df_i,
            freq=freq,
            n_lags=n_lags,
            n_forecasts=n_forecasts,
            config_missing=config_missing,
            config_regressors=config_regressors,
            config_lagged_regressors=config_lagged_regressors,
            config_events=config_events,
            config_seasonality=config_seasonality,
            predicting=predicting,
        ).copy(deep=True)
        df_handled_missing_aux["ID"] = df_name
        df_handled_missing = pd.concat((df_handled_missing, df_handled_missing_aux), ignore_index=True)
    return df_handled_missing


def _handle_missing_data_single_id(
    df: pd.DataFrame,
    freq: Optional[str],
    n_lags: int,
    n_forecasts: int,
    config_missing,
    config_regressors: Optional[ConfigFutureRegressors],
    config_lagged_regressors: Optional[ConfigLaggedRegressors],
    config_events: Optional[ConfigEvents],
    config_seasonality: Optional[ConfigSeasonality],
    predicting: bool = False,
) -> pd.DataFrame:
    """
    Checks and normalizes new data

    Data is also auto-imputed, unless impute_missing is set to ``False``.

        Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing columns 'ds', 'y', and optionally 'ID' of a single ID.
    freq : str
            data step sizes. Frequency of data recording,

            Note
            ----
            Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto`` (default) to automatically set frequency.
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
    # Receives df with single ID column
    assert len(df["ID"].unique()) == 1
    if n_lags == 0 and not predicting:
        # we can drop rows with NA in y
        sum_na = sum(df["y"].isna())
        if sum_na > 0:
            df = df[df["y"].notna()]
            log.info(f"dropped {sum_na} NAN row in 'y'")

    # add missing dates for autoregression modelling
    if n_lags > 0:
        df, missing_dates = df_utils.add_missing_dates_nan(df, freq=freq)
        if missing_dates > 0:
            if config_missing.impute_missing:
                log.info(f"{missing_dates} missing dates added.")
            # FIX Issue#52
            # Comment error raising to allow missing data for autoregression flow.
            # else:
            #     raise ValueError(f"{missing_dates} missing dates found. Please preprocess data manually or set impute_missing to True.")
            # END FIX

    if config_regressors is not None:
        # if future regressors, check that they are not nan at end, else drop
        # we ignore missing events, as those will be filled in with zeros.
        reg_nan_at_end = 0
        for col, regressor in config_regressors.items():
            # check for completeness of the regressor values
            col_nan_at_end = 0
            while len(df) > col_nan_at_end and df[col].isnull().iloc[-(1 + col_nan_at_end)]:
                col_nan_at_end += 1
            reg_nan_at_end = max(reg_nan_at_end, col_nan_at_end)
        if reg_nan_at_end > 0:
            # drop rows at end due to missing future regressors
            df = df[:-reg_nan_at_end]
            log.info(f"Dropped {reg_nan_at_end} rows at end due to missing future regressor values.")

    df_end_to_append = None
    nan_at_end = 0
    while len(df) > nan_at_end and df["y"].isnull().iloc[-(1 + nan_at_end)]:
        nan_at_end += 1
    if nan_at_end > 0:
        if predicting:
            # allow nans at end - will re-add at end
            if n_forecasts > 1 and n_forecasts < nan_at_end:
                # check that not more than n_forecasts nans, else drop surplus
                df = df[: -(nan_at_end - n_forecasts)]
                # correct new length:
                nan_at_end = n_forecasts
                log.info(
                    "Detected y to have more NaN values than n_forecast can predict. "
                    f"Dropped {nan_at_end - n_forecasts} rows at end."
                )
            df_end_to_append = df[-nan_at_end:]
            df = df[:-nan_at_end]
        else:
            # training - drop nans at end
            df = df[:-nan_at_end]
            log.info(
                f"Dropped {nan_at_end} consecutive nans at end. "
                "Training data can only be imputed up to last observation."
            )

    # impute missing values
    data_columns = []
    if n_lags > 0:
        data_columns.append("y")
    if config_lagged_regressors is not None:
        data_columns.extend(config_lagged_regressors.keys())
    if config_regressors is not None:
        data_columns.extend(config_regressors.keys())
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
        sum_na = sum(df[column].isnull())
        if sum_na > 0:
            log.warning(f"{sum_na} missing values in column {column} were detected in total. ")
            if config_missing.impute_missing:
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
                        f"More than {2 * config_missing.impute_linear + config_missing.impute_rolling} consecutive missing values encountered in column {column}. "
                        f"{remaining_na} NA remain after auto-imputation. "
                    )
            # FIX Issue#52
            # Comment error raising to allow missing data for autoregression flow.
            # else:  # fail because set to not impute missing
            #    raise ValueError(
            #        "Missing values found. " "Please preprocess data manually or set impute_missing to True."
            #    )
            # END FIX
    if df_end_to_append is not None:
        df = pd.concat([df, df_end_to_append])
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
        predict_steps=model.predict_steps,
        config_seasonality=model.config_seasonality,
        config_events=model.config_events,
        config_country_holidays=model.config_country_holidays,
        config_lagged_regressors=model.config_lagged_regressors,
        config_regressors=model.config_regressors,
        config_missing=model.config_missing,
        prediction_frequency=prediction_frequency,
    )
