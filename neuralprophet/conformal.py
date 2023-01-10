from dataclasses import dataclass
from typing import List, Optional

import matplotlib
import numpy as np
import pandas as pd

from neuralprophet.plot_forecast_matplotlib import plot_nonconformity_scores
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
    quantiles : list
        optional, list of quantiles for quantile regression uncertainty estimate

    """

    alpha: float
    method: str
    quantiles: Optional[List[float]] = None

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
        # conformalize
        self.noncon_scores = self._get_nonconformity_scores(df_cal)
        self.q_hat = self._get_q_hat(df_cal)
        df["qhat1"] = self.q_hat
        if self.method == "naive":
            df["yhat1 - qhat1"] = df["yhat1"] - self.q_hat
            df["yhat1 + qhat1"] = df["yhat1"] + self.q_hat
        elif self.method == "cqr":
            quantile_hi = str(max(self.quantiles) * 100)
            quantile_lo = str(min(self.quantiles) * 100)
            df[f"yhat1 {quantile_hi}% - qhat1"] = df[f"yhat1 {quantile_hi}%"] - self.q_hat
            df[f"yhat1 {quantile_hi}% + qhat1"] = df[f"yhat1 {quantile_hi}%"] + self.q_hat
            df[f"yhat1 {quantile_lo}% - qhat1"] = df[f"yhat1 {quantile_lo}%"] - self.q_hat
            df[f"yhat1 {quantile_lo}% + qhat1"] = df[f"yhat1 {quantile_lo}%"] + self.q_hat
        else:
            raise ValueError(
                f"Unknown conformal prediction method '{self.method}'. Please input either 'naive' or 'cqr'."
            )

        return df

    def _get_nonconformity_scores(self, df_cal: pd.DataFrame) -> np.ndarray:
        """Get the nonconformity scores using the given conformal prediction technique.

        Parameters
        ----------
            df_cal : pd.DataFrame
                calibration dataframe

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
                if row[f"yhat1 {quantile_lo}%"] is None or row[f"yhat1 {quantile_hi}%"] is None
                else [
                    max(
                        row[f"yhat1 {quantile_lo}%"] - row["y"],
                        row["y"] - row[f"yhat1 {quantile_hi}%"],
                    ),
                    0 if row[f"yhat1 {quantile_lo}%"] - row["y"] > row["y"] - row[f"yhat1 {quantile_hi}%"] else 1,
                ]
            )
            scores_df = df_cal.apply(cqr_scoring_func, axis=1, result_type="expand")
            scores_df.columns = ["scores", "arg"]
            noncon_scores = scores_df["scores"].values
        else:  # self.method == "naive"
            # Naive nonconformity scoring function
            noncon_scores = abs(df_cal["y"] - df_cal["yhat1"]).values
        # Remove NaN values
        noncon_scores = noncon_scores[~pd.isnull(noncon_scores)]
        # Sort
        noncon_scores.sort()

        return noncon_scores

    def _get_q_hat(self, df_cal: pd.DataFrame) -> float:
        """Get the q_hat that is derived from the nonconformity scores.

        Parameters
        ----------
            df_cal : pd.DataFrame
                calibration dataframe

            Returns
            -------
                float
                    q_hat value, or the one-sided prediction interval width

        """
        # Get the q-hat index and value
        q_hat_idx = int(len(self.noncon_scores) * self.alpha)
        q_hat = self.noncon_scores[-q_hat_idx]

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
            fig = plot_nonconformity_scores_plotly(self.noncon_scores, self.alpha, self.q_hat, method)
        elif plotting_backend == "matplotlib":
            fig = plot_nonconformity_scores(self.noncon_scores, self.alpha, self.q_hat, method)
        if plotting_backend in ["matplotlib", "plotly"] and matplotlib.is_interactive():
            fig.show()
