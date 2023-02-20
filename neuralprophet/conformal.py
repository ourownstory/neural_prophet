from dataclasses import dataclass
from typing import Any, List

import matplotlib
import numpy as np
import pandas as pd

from neuralprophet.plot_forecast_matplotlib import plot_interval_width_per_timestep, plot_nonconformity_scores
from neuralprophet.plot_forecast_plotly import (
    plot_interval_width_per_timestep as plot_interval_width_per_timestep_plotly,
)
from neuralprophet.plot_forecast_plotly import plot_nonconformity_scores as plot_nonconformity_scores_plotly


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
            df[f"qhat{step_number}"] = q_hat
            if self.method == "naive":
                df[f"yhat{step_number} - qhat{step_number}"] = df[f"yhat{step_number}"] - q_hat
                df[f"yhat{step_number} + qhat{step_number}"] = df[f"yhat{step_number}"] + q_hat
            elif self.method == "cqr":
                quantile_hi = str(max(self.quantiles) * 100)
                quantile_lo = str(min(self.quantiles) * 100)
                df[f"yhat{step_number} {quantile_hi}% - qhat{step_number}"] = (
                    df[f"yhat{step_number} {quantile_hi}%"] - q_hat
                )
                df[f"yhat{step_number} {quantile_hi}% + qhat{step_number}"] = (
                    df[f"yhat{step_number} {quantile_hi}%"] + q_hat
                )
                df[f"yhat{step_number} {quantile_lo}% - qhat{step_number}"] = (
                    df[f"yhat{step_number} {quantile_lo}%"] - q_hat
                )
                df[f"yhat{step_number} {quantile_lo}% + qhat{step_number}"] = (
                    df[f"yhat{step_number} {quantile_lo}%"] + q_hat
                )
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
        if self.method == "cqr":
            # CQR nonconformity scoring function
            quantile_hi = str(max(self.quantiles) * 100)
            quantile_lo = str(min(self.quantiles) * 100)
            cqr_scoring_func = (
                lambda row: [None, None]
                if row[f"yhat{step_number} {quantile_lo}%"] is None or row[f"yhat{step_number} {quantile_hi}%"] is None
                else [
                    max(
                        row[f"yhat{step_number} {quantile_lo}%"] - row["y"],
                        row["y"] - row[f"yhat{step_number} {quantile_hi}%"],
                    ),
                    0
                    if row[f"yhat{step_number} {quantile_lo}%"] - row["y"]
                    > row["y"] - row[f"yhat{step_number} {quantile_hi}%"]
                    else 1,
                ]
            )
            scores_df = df_cal.apply(cqr_scoring_func, axis=1, result_type="expand")
            scores_df.columns = ["scores", "arg"]
            noncon_scores = scores_df["scores"].values
        else:  # self.method == "naive"
            # Naive nonconformity scoring function
            noncon_scores = abs(df_cal["y"] - df_cal[f"yhat{step_number}"]).values
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
        q_hat_idx = int(len(noncon_scores) * self.alpha)
        q_hat = noncon_scores[-q_hat_idx]

        return q_hat

    def plot(self, plotting_backend: str):
        """Apply a given conformal prediction technique to get the uncertainty prediction intervals (or q-hats).

        Parameters
        ----------
            plotting_backend : str
                specifies the plotting backend for the nonconformity scores plot, if any

                Options
                    * ``matplotlib``: Use matplotlib backend for plotting
                    * ``plotly``: Use the plotly backend for plotting

        """
        method = self.method.upper() if "cqr" in self.method.lower() else self.method.title()
        if plotting_backend == "plotly":
            if self.n_forecasts == 1:
                # includes nonconformity scores of the first timestep
                fig = plot_nonconformity_scores_plotly(self.noncon_scores, self.alpha, self.q_hats[0], method)
            else:
                fig = plot_interval_width_per_timestep_plotly(self.q_hats, method)
        elif plotting_backend == "matplotlib":
            if self.n_forecasts == 1:
                # includes nonconformity scores of the first timestep
                fig = plot_nonconformity_scores(self.noncon_scores, self.alpha, self.q_hats[0], method)
            else:
                fig = plot_interval_width_per_timestep(self.q_hats, method)
        else:
            raise ValueError(
                f"Unknown plotting backend '{plotting_backend}'. Please input either 'matplotlib' or 'plotly'."
            )
        if plotting_backend in ["matplotlib", "plotly"] and matplotlib.is_interactive():
            fig.show()

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate conformal prediction on test dataframe.

        Parameters
        ----------
            df : pd.DataFrame
                test dataframe with the conformal prediction intervals

        Returns
        -------
            pd.DataFrame
                table containing the two evaluation metrics interval_width and miscoverage_rate for each predicted interval
        """
        eval_df = pd.DataFrame()
        for step_number in range(1, self.n_forecasts + 1):
            q_hat = self.q_hats[step_number - 1]
            if self.method == "naive":
                # Interval width (efficiency metric)
                interval_width = q_hat * 2
                # Miscoverage rate (validity metric)
                n_covered = df.apply(
                    lambda row: bool(
                        row[f"yhat{step_number} - qhat{step_number}"]
                        <= row["y"]
                        <= row[f"yhat{step_number} + qhat{step_number}"]
                    ),
                    axis=1,
                )
                coverage_rate = n_covered.sum() / len(df)
                miscoverage_rate = 1 - coverage_rate
            elif self.method == "cqr":
                quantile_hi = str(max(self.quantiles) * 100)
                quantile_lo = str(min(self.quantiles) * 100)
                # Interval width (efficiency metric)
                quantile_lo_mean = df[f"yhat{step_number}"].mean() - df[f"yhat{step_number} {quantile_lo}%"].mean()
                quantile_hi_mean = df[f"yhat{step_number} {quantile_hi}%"].mean() - df[f"yhat{step_number}"].mean()
                interval_width = quantile_lo_mean + quantile_hi_mean + q_hat * 2
                # Miscoverage rate (validity metric)
                n_covered = df.apply(
                    lambda row: bool(
                        row[f"yhat{step_number} {quantile_lo}% - qhat{step_number}"]
                        <= row["y"]
                        <= row[f"yhat{step_number} {quantile_hi}% + qhat{step_number}"]
                    ),
                    axis=1,
                )
                coverage_rate = n_covered.sum() / len(df)
                miscoverage_rate = 1 - coverage_rate
            else:
                raise ValueError(
                    f"Unknown conformal prediction method '{self.method}'. Please input either 'naive' or 'cqr'."
                )
            # Construct evaluation dataframe using q-hat, interval width, and miscoverage rate
            row = [q_hat, interval_width, miscoverage_rate]
            eval_cols = [f"qhat{step_number}", "interval_width", "miscoverage_rate"]
            eval_df = pd.DataFrame([row], columns=pd.MultiIndex.from_product([[f"yhat{step_number}"], eval_cols]))

        return eval_df
