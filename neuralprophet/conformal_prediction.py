import pandas as pd

from neuralprophet.plot_forecast import plot_nonconformity_scores

import logging
log = logging.getLogger("NP.cp")
log.setLevel("DEBUG")
log.parent.setLevel("WARNING")


def conformalize(df_cal, alpha, method, quantiles):
    # get non-conformity scores and sort them
    log.info("CHECK3")
    print("CHECK3")
    q_hats = []
    noncon_scores_list, quantile_hi, quantile_lo = _get_nonconformity_scores(df_cal, method, quantiles)

    for noncon_scores in noncon_scores_list:
        log.info("CHECK7")
        print("CHECK7")
        noncon_scores = noncon_scores[~pd.isnull(noncon_scores)]  # remove NaN values
        noncon_scores.sort()
        log.info("CHECK8")
        print("CHECK8")
        # get the q-hat index and value
        q_hat_idx = int(len(noncon_scores) * alpha)
        q_hat = noncon_scores[-q_hat_idx]
        q_hats.append(q_hat)
        log.info("CHECK9")
        print("CHECK9")
        method = method.upper() if "cqr" in method.lower() else method.title()
        plot_nonconformity_scores(noncon_scores, q_hat, method)
        log.info("CHECK10")
        print("CHECK10")

    return q_hats, quantile_hi, quantile_lo


def _get_nonconformity_scores(df, method, quantiles):
    log.info("CHECK4")
    print("CHECK4")
    quantile_hi = None
    quantile_lo = None

    if method == "cqr":
        # CQR nonconformity scoring function
        log.info("CHECK5")
        print("CHECK5")
        quantile_hi = str(max(quantiles) * 100)
        quantile_lo = str(min(quantiles) * 100)
        log.info("CHECK5.1")
        print("CHECK5.1")
        cqr_scoring_func = (
            lambda row: [None, None]
            if row[f"yhat1 {quantile_lo}%"] is None or row[f"yhat1 {quantile_hi}%"] is None
            else [
                max(row[f"yhat1 {quantile_lo}%"] - row["y"], row["y"] - row[f"yhat1 {quantile_hi}%"]),
                0 if row[f"yhat1 {quantile_lo}%"] - row["y"] > row["y"] - row[f"yhat1 {quantile_hi}%"] else 1,
            ]
        )
        scores_df = df.apply(cqr_scoring_func, axis=1, result_type="expand")
        log.info("CHECK5.2")
        print("CHECK5.2")
        scores_df.columns = ["scores", "arg"]
        log.info("CHECK5.3")
        print("CHECK5.3")
        scores_list = [scores_df["scores"].values]
    else:  # method == "naive"
        # Naive nonconformity scoring function
        log.info("CHECK5")
        print("CHECK5")
        scores_list = [abs(df["residual1"]).values]

    log.info("CHECK6")
    print("CHECK6")
    return scores_list, quantile_hi, quantile_lo
