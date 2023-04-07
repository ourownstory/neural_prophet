import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

from neuralprophet.plot_forecast_matplotlib import plot_interval_width_per_timestep, plot_nonconformity_scores
from neuralprophet.plot_forecast_plotly import (
    plot_interval_width_per_timestep as plot_interval_width_per_timestep_plotly,
)
from neuralprophet.plot_forecast_plotly import plot_nonconformity_scores as plot_nonconformity_scores_plotly
from neuralprophet.plot_utils import log_warning_deprecation_plotly, select_plotting_backend


@dataclass
class Conformal:
    """Conformal prediction dataclass

    Parameters
    ----------
    alpha : float
        user-specified significance level of the prediction interval
    method : str
        name of conformal prediction technique used

        Options
            * ``naive``: Naive or Absolute Residual
            * ``cqr``: Conformalized Quantile Regression
    n_forecasts : int
        optional, number of steps ahead of prediction time step to forecast
    quantiles : list
        optional, list of quantiles for quantile regression uncertainty estimate

    """

    alpha: float
    method: str
    n_forecasts: int
    quantiles: List[float]

    def predict(self, df: pd.DataFrame, df_cal: pd.DataFrame) -> pd.DataFrame:
        """Apply a given conformal prediction technique to get the uncertainty prediction intervals (or q-hat) for test dataframe.

        Parameters
        ----------
            df : pd.DataFrame
                test dataframe
            df_cal : pd.DataFrame
                calibration dataframe

            Returns
            -------
                pd.DataFrame
                    test dataframe with uncertainty prediction intervals

        """
        self.q_hats = []
        for step_number in range(1, self.n_forecasts + 1):
            # conformalize
            noncon_scores = self._get_nonconformity_scores(df_cal, step_number)
            q_hat = self._get_q_hat(df_cal, noncon_scores)
            q_hat_col = f"qhat{step_number}"
            y_hat_col = f"yhat{step_number}"
            df[q_hat_col] = q_hat
            if self.method == "naive":
                df[f"{y_hat_col} - {q_hat_col}"] = df[y_hat_col] - q_hat
                df[f"{y_hat_col} + {q_hat_col}"] = df[y_hat_col] + q_hat
            elif self.method == "cqr":
                quantile_lo = str(min(self.quantiles) * 100)
                quantile_hi = str(max(self.quantiles) * 100)
                quantile_lo_col = f"{y_hat_col} {quantile_lo}%"
                quantile_hi_col = f"{y_hat_col} {quantile_hi}%"
                df[f"{quantile_lo_col} - {q_hat_col}"] = df[quantile_lo_col] - q_hat
                df[f"{quantile_lo_col} + {q_hat_col}"] = df[quantile_lo_col] + q_hat
                df[f"{quantile_hi_col} - {q_hat_col}"] = df[quantile_hi_col] - q_hat
                df[f"{quantile_hi_col} + {q_hat_col}"] = df[quantile_hi_col] + q_hat
            else:
                raise ValueError(
                    f"Unknown conformal prediction method '{self.method}'. Please input either 'naive' or 'cqr'."
                )
            if step_number == 1:
                # save nonconformity scores of the first timestep
                self.noncon_scores = noncon_scores
            self.q_hats.append(q_hat)

        return df

    def _get_nonconformity_scores(self, df_cal: pd.DataFrame, step_number: int) -> np.ndarray:
        """Get the nonconformity scores using the given conformal prediction technique.

        Parameters
        ----------
            df_cal : pd.DataFrame
                calibration dataframe
            step_number : int
                i-th step ahead forecast

            Returns
            -------
                np.ndarray
                    nonconformity scores from the calibration datapoints

        """
        y_hat_col = f"yhat{step_number}"
        if self.method == "cqr":
            # CQR nonconformity scoring function
            quantile_lo = str(min(self.quantiles) * 100)
            quantile_hi = str(max(self.quantiles) * 100)
            quantile_lo_col = f"{y_hat_col} {quantile_lo}%"
            quantile_hi_col = f"{y_hat_col} {quantile_hi}%"
            cqr_scoring_func = (
                lambda row: [None, None]
                if row[quantile_lo_col] is None or row[quantile_hi_col] is None
                else [
                    max(
                        row[quantile_lo_col] - row["y"],
                        row["y"] - row[quantile_hi_col],
                    ),
                    0 if row[quantile_lo_col] - row["y"] > row["y"] - row[quantile_hi_col] else 1,
                ]
            )
            scores_df = df_cal.apply(cqr_scoring_func, axis=1, result_type="expand")
            scores_df.columns = ["scores", "arg"]
            noncon_scores = scores_df["scores"].values
        else:  # self.method == "naive"
            # Naive nonconformity scoring function
            noncon_scores = abs(df_cal["y"] - df_cal[y_hat_col]).values
        # Remove NaN values
        noncon_scores: Any = noncon_scores[~pd.isnull(noncon_scores)]
        # Sort
        noncon_scores.sort()

        return noncon_scores

    def _get_q_hat(self, df_cal: pd.DataFrame, noncon_scores: np.ndarray) -> float:
        """Get the q_hat that is derived from the nonconformity scores.

        Parameters
        ----------
            df_cal : pd.DataFrame
                calibration dataframe
            noncon_scores : np.ndarray
                nonconformity scores

            Returns
            -------
                float
                    q_hat value, or the one-sided prediction interval width

        """
        # Get the q-hat index and value
        q_hat_idx = round(len(noncon_scores) * self.alpha)
        q_hat = noncon_scores[-q_hat_idx]

        return q_hat

    def plot(self, plotting_backend=None):
        """Apply a given conformal prediction technique to get the uncertainty prediction intervals (or q-hats).

        Parameters
        ----------
            plotting_backend : str
                specifies the plotting backend for the nonconformity scores plot, if any

                Options
                * ``plotly-resampler``: Use the plotly backend for plotting in resample mode. This mode uses the
                    plotly-resampler package to accelerate visualizing large data by resampling it. For some
                    environments (colab, pycharm interpreter) plotly-resampler might not properly vizualise the figures.
                    In this case, consider switching to 'plotly-auto'.
                * ``plotly``: Use the plotly backend for plotting
                * ``matplotlib``: use matplotlib for plotting
                * (default) None: Plotting backend ist set automatically. Use plotly with resampling for jupyterlab
                    notebooks and vscode notebooks. Automatically switch to plotly without resampling for all other
                    environments.

        """
        method = self.method.upper() if "cqr" in self.method.lower() else self.method.title()
        # Check whether a local or global plotting backend is set.
        plotting_backend = select_plotting_backend(model=self, plotting_backend=plotting_backend)

        log_warning_deprecation_plotly(plotting_backend)
        if plotting_backend.startswith("plotly"):
            if self.n_forecasts == 1:
                # includes nonconformity scores of the first timestep
                fig = plot_nonconformity_scores_plotly(
                    self.noncon_scores,
                    self.alpha,
                    self.q_hats[0],
                    method,
                    resampler_active=plotting_backend == "plotly-resampler",
                )
            else:
                fig = plot_interval_width_per_timestep_plotly(self.q_hats, method, resampler_active=False)
        else:
            if self.n_forecasts == 1:
                # includes nonconformity scores of the first timestep
                fig = plot_nonconformity_scores(self.noncon_scores, self.alpha, self.q_hats[0], method)
            else:
                fig = plot_interval_width_per_timestep(self.q_hats, method)
        if plotting_backend in ["matplotlib", "plotly", "plotly-resampler"] and matplotlib.is_interactive():
            fig


def uncertainty_evaluate(df_forecast: pd.DataFrame) -> pd.DataFrame:
    """Evaluate conformal prediction on test dataframe.

    Parameters
    ----------
        df_forecast : pd.DataFrame
            forecast dataframe with the conformal prediction intervals

    Returns
    -------
        pd.DataFrame
            table containing evaluation metrics such as interval_width and miscoverage_rate
    """
    # Remove beginning rows used as lagged regressors (if any), or future dataframes without y-values
    # therefore, this ensures that all forecast rows for evaluation contains both y and y-hat
    df_forecast_eval = df_forecast.dropna(subset=["y", "yhat1"]).reset_index(drop=True)
    # Get evaluation params
    method, n_forecasts, quantile_lo, quantile_hi = _infer_evaluate_params_from_dataset(df_forecast_eval)
    df_eval = pd.DataFrame()
    # Begin conformal evaluation steps
    for step_number in range(1, n_forecasts + 1):
        q_hat_col = f"qhat{step_number}"
        y_hat_col = f"yhat{step_number}"
        q_hat = df_forecast_eval.iloc[0][q_hat_col]
        # QR Interval Evaluation (if quantiles lo & hi both exist)
        if quantile_lo and quantile_hi:
            quantile_lo_col = f"{y_hat_col} {quantile_lo}%"
            quantile_hi_col = f"{y_hat_col} {quantile_hi}%"
            # Get QR evaluation metrics
            interval_width, miscoverage_rate = _get_evaluate_metrics_from_dataset(
                df_forecast_eval, quantile_lo_col, quantile_hi_col
            )
            # Construct row dataframe with current timestep using its q-hat, interval width, and miscoverage rate
            col_names = ["interval_width", "miscoverage_rate"]
            row = [interval_width, miscoverage_rate]
            df_row = pd.DataFrame([row], columns=pd.MultiIndex.from_product([[y_hat_col], ["qr"], col_names]))
            # Add row dataframe to overall evaluation dataframe with all forecasted timesteps
            df_eval = pd.concat([df_eval, df_row], axis=1)
        # Naive CP Interval Evaluation
        if method == "naive":
            quantile_lo_col = f"{y_hat_col} - {q_hat_col}"
            quantile_hi_col = f"{y_hat_col} + {q_hat_col}"
        # CQR Interval Evaluation
        elif method == "cqr":
            quantile_lo_col = f"{y_hat_col} {quantile_lo}% - {q_hat_col}"
            quantile_hi_col = f"{y_hat_col} {quantile_hi}% + {q_hat_col}"
        else:
            raise ValueError(f"Unknown conformal prediction method '{method}'. Please input either 'naive' or 'cqr'.")
        # Get CP evaluation metrics
        interval_width, miscoverage_rate = _get_evaluate_metrics_from_dataset(
            df_forecast_eval, quantile_lo_col, quantile_hi_col
        )
        # Construct row dataframe with current timestep using its q-hat, interval width, and miscoverage rate
        col_names = [f"qhat{step_number}", "interval_width", "miscoverage_rate"]
        row = [q_hat, interval_width, miscoverage_rate]
        df_row = pd.DataFrame([row], columns=pd.MultiIndex.from_product([[y_hat_col], [method], col_names]))
        # Add row dataframe to overall evaluation dataframe with all forecasted timesteps
        df_eval = pd.concat([df_eval, df_row], axis=1)

    return df_eval


def _infer_evaluate_params_from_dataset(
    df_forecast_eval: pd.DataFrame,
) -> Tuple[str, int, Optional[str], Optional[str]]:
    """Infers evaluation parameters based on the evaluation dataframe columns.

    Parameters
    ----------
        df_forecast_eval : pd.DataFrame
            forecast dataframe with the conformal prediction intervals

    Returns
    -------
        str, int, Optional[str], Optional[str]
            parameters to evaluate conformal prediction, only cqr outputs quantile_lo and quantile_hi
    """
    # Get n_forecasts
    qhat_col = [col for col in df_forecast_eval.columns if col[:4] == "qhat"]
    n_forecasts = int(qhat_col[-1].replace("qhat", ""))
    # Extract conformal prediction forecast column(s)
    cp_pattern = "yhat1\\ (.*)?\\%\\ \\+\\ qhat1"
    cp_col = [col for col in df_forecast_eval if re.compile(cp_pattern).match(col)]
    # Get Naive method if only "yhat1 + qhat1" exist and CQR if "yhat1 {quantile} + qhat1" for both lo & hi exist
    method = "cqr" if len(cp_col) == 2 else "naive"
    # Extract quantile regression forecast column(s)
    qr_pattern = "yhat1\\ (.*)?\\%$"
    qr_col = [col for col in df_forecast_eval if re.compile(qr_pattern).match(col)]
    # Get quantile lo & hi if QR exists else set both to None
    if len(qr_col) == 2:
        quantile_lo = re.findall(qr_pattern, qr_col[0])[0]
        quantile_hi = re.findall(qr_pattern, qr_col[1])[0]
    else:
        quantile_lo = None
        quantile_hi = None

    return method, n_forecasts, quantile_lo, quantile_hi


def _get_evaluate_metrics_from_dataset(
    df_forecast_eval: pd.DataFrame,
    quantile_lo_col: str,
    quantile_hi_col: str,
) -> Tuple[float, float]:
    """Infers evaluation parameters based on the evaluation dataframe columns.

    Parameters
    ----------
        df_forecast_eval : pd.DataFrame
            forecast dataframe with the conformal prediction intervals

    Returns
    -------
        float, float
            conformal prediction evaluation metrics
    """
    # Interval width (efficiency metric)
    quantile_lo_mean = df_forecast_eval[quantile_lo_col].mean()
    quantile_hi_mean = df_forecast_eval[quantile_hi_col].mean()
    interval_width = quantile_hi_mean - quantile_lo_mean
    # Miscoverage rate (validity metric)
    n_covered = df_forecast_eval.apply(
        lambda row: bool(row[quantile_lo_col] <= row["y"] <= row[quantile_hi_col]),
        axis=1,
    )
    coverage_rate = n_covered.sum() / len(df_forecast_eval)
    miscoverage_rate = 1 - coverage_rate

    return interval_width, miscoverage_rate
