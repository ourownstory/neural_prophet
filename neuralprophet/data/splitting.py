import logging

import pandas as pd

from neuralprophet import df_utils

log = logging.getLogger("NP.data.splitting")


def _maybe_extend_df(self, df):
    # Receives df with ID column
    periods_add = {}
    extended_df = pd.DataFrame()
    for df_name, df_i in df.groupby("ID"):
        _ = df_utils.infer_frequency(df_i, n_lags=self.max_lags, freq=self.data_freq)
        # to get all forecasteable values with df given, maybe extend into future:
        periods_add[df_name] = self._get_maybe_extend_periods(df_i)
        if periods_add[df_name] > 0:
            # This does not include future regressors or events.
            # periods should be 0 if those are configured.
            last_date = pd.to_datetime(df_i["ds"].copy(deep=True)).sort_values().max()
            future_df = df_utils.make_future_df(
                df_columns=df_i.columns,
                last_date=last_date,
                periods=periods_add[df_name],
                freq=self.data_freq,
                config_events=self.config_events,
            )
            future_df["ID"] = df_name
            df_i = pd.concat([df_i, future_df])
            df_i.reset_index(drop=True, inplace=True)
        extended_df = pd.concat((extended_df, df_i.copy(deep=True)), ignore_index=True)
    return extended_df, periods_add


def _get_maybe_extend_periods(self, df):
    # Receives df with single ID column
    assert len(df["ID"].unique()) == 1
    periods_add = 0
    nan_at_end = 0
    while len(df) > nan_at_end and df["y"].isnull().iloc[-(1 + nan_at_end)]:
        nan_at_end += 1
    if self.max_lags > 0:
        if self.config_regressors is None:
            # if dataframe has already been extended into future,
            # don't extend beyond n_forecasts.
            periods_add = max(0, self.n_forecasts - nan_at_end)
        else:
            # can not extend as we lack future regressor values.
            periods_add = 0
    return periods_add


def _make_future_dataframe(self, df, events_df, regressors_df, periods, n_historic_predictions):
    # Receives df with single ID column
    assert len(df["ID"].unique()) == 1
    if periods == 0 and n_historic_predictions is True:
        log.warning("Not extending df into future as no periods specified." "You can call predict directly instead.")
    df = df.copy(deep=True)
    _ = df_utils.infer_frequency(df, n_lags=self.max_lags, freq=self.data_freq)
    last_date = pd.to_datetime(df["ds"].copy(deep=True).dropna()).sort_values().max()
    if events_df is not None:
        events_df = events_df.copy(deep=True).reset_index(drop=True)
    if regressors_df is not None:
        regressors_df = regressors_df.copy(deep=True).reset_index(drop=True)
    if periods is None:
        periods = 1 if self.max_lags == 0 else self.n_forecasts
    else:
        assert periods >= 0

    if isinstance(n_historic_predictions, bool):
        if n_historic_predictions:
            n_historic_predictions = len(df) - self.max_lags
        else:
            n_historic_predictions = 0
    elif not isinstance(n_historic_predictions, int):
        log.error("non-integer value for n_historic_predictions set to zero.")
        n_historic_predictions = 0

    if periods == 0 and n_historic_predictions == 0:
        raise ValueError("Set either history or future to contain more than zero values.")

    # check for external regressors known in future
    if self.config_regressors is not None and periods > 0:
        if regressors_df is None:
            raise ValueError("Future values of all user specified regressors not provided")
        else:
            for regressor in self.config_regressors.keys():
                if regressor not in regressors_df.columns:
                    raise ValueError(f"Future values of user specified regressor {regressor} not provided")

    if len(df) < self.max_lags:
        raise ValueError(
            "Insufficient input data for a prediction."
            "Please supply historic observations (number of rows) of at least max_lags (max of number of n_lags)."
        )
    elif len(df) < self.max_lags + n_historic_predictions:
        log.warning(
            f"Insufficient data for {n_historic_predictions} historic forecasts, reduced to {len(df) - self.max_lags}."
        )
        n_historic_predictions = len(df) - self.max_lags
    if (n_historic_predictions + self.max_lags) == 0:
        df = pd.DataFrame(columns=df.columns)
    else:
        df = df[-(self.max_lags + n_historic_predictions) :]
        nan_at_end = 0
        while len(df) > nan_at_end and df["y"].isnull().iloc[-(1 + nan_at_end)]:
            nan_at_end += 1
        if nan_at_end > 0:
            if self.max_lags > 0 and (nan_at_end + 1) >= self.max_lags:
                raise ValueError(
                    f"{nan_at_end + 1} missing values were detected at the end of df before df was extended into the future. "
                    "Please make sure there are no NaN values at the end of df."
                )
            df["y"].iloc[-(nan_at_end + 1) :].ffill(inplace=True)
            log.warning(
                f"{nan_at_end + 1} missing values were forward-filled at the end of df before df was extended into the future. "
                "Please make sure there are no NaN values at the end of df."
            )

    if len(df) > 0:
        if len(df.columns) == 1 and "ds" in df:
            assert self.max_lags == 0
            df = self._check_dataframe(df, check_y=False, exogenous=False)
        else:
            df = self._check_dataframe(df, check_y=self.max_lags > 0, exogenous=True)
    # future data
    # check for external events known in future
    if self.config_events is not None and periods > 0 and events_df is None:
        log.warning(
            "Future values not supplied for user specified events. "
            "All events being treated as not occurring in future"
        )

    if self.max_lags > 0:
        if periods > 0 and periods != self.n_forecasts:
            periods = self.n_forecasts
            log.warning(f"Number of forecast steps is defined by n_forecasts. Adjusted to {self.n_forecasts}.")

    if periods > 0:
        future_df = df_utils.make_future_df(
            df_columns=df.columns,
            last_date=last_date,
            periods=periods,
            freq=self.data_freq,
            config_events=self.config_events,
            events_df=events_df,
            config_regressors=self.config_regressors,
            regressors_df=regressors_df,
        )
        if len(df) > 0:
            df = pd.concat([df, future_df])
        else:
            df = future_df
    df = df.reset_index(drop=True)
    self.predict_steps = periods
    return df
