import matplotlib
import pandas as pd

from neuralprophet.plot_forecast_matplotlib import plot_nonconformity_scores
from neuralprophet.plot_forecast_plotly import plot_nonconformity_scores as plot_nonconformity_scores_plotly


def conformalize(df_cal, alpha, method, quantiles, plotting_backend):
    """Apply a given conformal prediction technique to get the uncertainty prediction intervals (or q-hats).

    Parameters
    ----------
        df_cal : pd.DataFrame
            calibration dataframe
        alpha : float
            user-specified significance level of the prediction interval
        method : str
            name of conformal prediction technique used

            Options
                * (default) ``naive``: Naive or Absolute Residual
                * ``cqr``: Conformalized Quantile Regression

        quantiles : list
            list of quantiles for quantile regression uncertainty estimate

        plotting_backend : str
            specifies the plotting backend for the nonconformity scores plot, if any

            Options
                * ``None``: No plotting is shown
                * ``plotly``: Use the plotly backend for plotting
                * ``matplotlib``: Use matplotlib backend for plotting
                * ``default`` (default): Use matplotlib backend for plotting

        Returns
        -------
            list
                uncertainty prediction intervals (or q-hats)

    """
    # get non-conformity scores and sort them
    q_hats = []
    noncon_scores_list = _get_nonconformity_scores(df_cal, method, quantiles)

    for noncon_scores in noncon_scores_list:
        noncon_scores = noncon_scores[~pd.isnull(noncon_scores)]  # remove NaN values
        noncon_scores.sort()
        # get the q-hat index and value
        q_hat_idx = int(len(noncon_scores) * alpha)
        q_hat = noncon_scores[-q_hat_idx]
        q_hats.append(q_hat)
        method = method.upper() if "cqr" in method.lower() else method.title()
        if plotting_backend == "plotly":
            fig = plot_nonconformity_scores_plotly(noncon_scores, alpha, q_hat, method)
        elif plotting_backend == "matplotlib":
            fig = plot_nonconformity_scores(noncon_scores, alpha, q_hat, method)
        if plotting_backend in ["matplotlib", "plotly"] and matplotlib.is_interactive():
            fig.show()

    return q_hats


def _get_nonconformity_scores(df, method, quantiles):
    """Get the nonconformity scores using the given conformal prediction technique.

    Parameters
    ----------
        df : pd.DataFrame
            calibration dataframe
        method : str
            name of conformal prediction technique used

            Options
                * (default) ``naive``: Naive or Absolute Residual
                * ``cqr``: Conformalized Quantile Regression

        quantiles : list
            list of quantiles for quantile regression uncertainty estimate

        Returns
        -------
            list
                nonconformity scores from the calibration datapoints

    """
    quantile_hi = None
    quantile_lo = None

    if method == "cqr":
        # CQR nonconformity scoring function
        quantile_hi = str(max(quantiles) * 100)
        quantile_lo = str(min(quantiles) * 100)
        cqr_scoring_func = (
            lambda row: [None, None]
            if row[f"yhat1 {quantile_lo}%"] is None or row[f"yhat1 {quantile_hi}%"] is None
            else [
                max(row[f"yhat1 {quantile_lo}%"] - row["y"], row["y"] - row[f"yhat1 {quantile_hi}%"]),
                0 if row[f"yhat1 {quantile_lo}%"] - row["y"] > row["y"] - row[f"yhat1 {quantile_hi}%"] else 1,
            ]
        )
        scores_df = df.apply(cqr_scoring_func, axis=1, result_type="expand")
        scores_df.columns = ["scores", "arg"]
        scores_list = [scores_df["scores"].values]
    else:  # method == "naive"
        # Naive nonconformity scoring function
        scores_list = [abs(df["y"] - df["yhat1"]).values]

    return scores_list
