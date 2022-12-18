# import logging
# from typing import Callable, List, Optional, Type, Union

import matplotlib
import pandas as pd

from neuralprophet.plot_forecast_matplotlib import plot_nonconformity_scores
from neuralprophet.plot_forecast_plotly import plot_nonconformity_scores as plot_nonconformity_scores_plotly

# log = logging.getLogger("NP.conformal")


class Conformal:
    """Conformal prediction class."""

    def __init__(self, alpha, method, quantiles=None):
        """
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
        self.alpha = alpha
        self.method = method
        self.quantiles = quantiles

    def predict(self, df, df_cal):
        # conformalize
        self.q_hat, self.noncon_scores = self._conformalize(df_cal)
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

    def _conformalize(self, df_cal):
        """Apply a given conformal prediction technique to get the uncertainty prediction intervals (or q-hats).

        Parameters
        ----------
            df_cal : pd.DataFrame
                calibration dataframe

            Returns
            -------
                list
                    uncertainty prediction intervals (or q-hats)

        """
        # get non-conformity scores and sort them
        noncon_scores = self._get_nonconformity_scores(df_cal)
        noncon_scores = noncon_scores[~pd.isnull(noncon_scores)]  # remove NaN values
        noncon_scores.sort()
        # get the q-hat index and value
        q_hat_idx = int(len(noncon_scores) * self.alpha)
        q_hat = noncon_scores[-q_hat_idx]

        return q_hat, noncon_scores

    def _get_nonconformity_scores(self, df_cal):
        """Get the nonconformity scores using the given conformal prediction technique.

        Parameters
        ----------
            df_cal : pd.DataFrame
                calibration dataframe
            method : str
                name of conformal prediction technique used

                Options
                    * (default) ``naive``: Naive or Absolute Residual
                    * ``cqr``: Conformalized Quantile Regression
            step_number : int
                i-th step ahead forecast to use for statistics and plotting
            quantiles : list
                list of quantiles for quantile regression uncertainty estimate

            Returns
            -------
                list
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
            scores = scores_df["scores"].values
        else:  # self.method == "naive"
            # Naive nonconformity scoring function
            scores = abs(df_cal["y"] - df_cal["yhat1"]).values

        return scores

    def plot(self, plotting_backend):
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
