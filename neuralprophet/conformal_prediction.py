import matplotlib
import pandas as pd

from neuralprophet.plot_forecast import plot_nonconformity_scores


def conformalize(df_cal, alpha, method, quantiles):
    # get non-conformity scores and sort them
    q_hats = []
    noncon_scores_list, quantile_hi, quantile_lo = _get_nonconformity_scores(df_cal, method, quantiles)

    for noncon_scores in noncon_scores_list:
        noncon_scores = noncon_scores[~pd.isnull(noncon_scores)]  # remove NaN values
        noncon_scores.sort()
        # get the q-hat index and value
        q_hat_idx = int(len(noncon_scores) * alpha)
        q_hat = noncon_scores[-q_hat_idx]
        q_hats.append(q_hat)
        method = method.upper() if "cqr" in method.lower() else method.title()
        fig = plot_nonconformity_scores(noncon_scores, q_hat, method)
        if matplotlib.is_interactive():
            fig.show()

    return q_hats, quantile_hi, quantile_lo


def _get_nonconformity_scores(df, method, quantiles):
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
        scores_list = [abs(df["residual1"]).values]

    return scores_list, quantile_hi, quantile_lo
