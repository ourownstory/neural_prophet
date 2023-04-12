import logging

import pandas as pd

from neuralprophet import df_utils

log = logging.getLogger("NP.data.transforming")


def _normalize(model, df):
    """Apply data scales.

    Applies data scaling factors to df using data_params.

    Parameters
    ----------
        df : pd.DataFrame
            dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data

    Returns
    -------
        df: pd.DataFrame, normalized
    """
    df, _, _, _ = df_utils.prep_or_copy_df(df)
    df_norm = pd.DataFrame()
    for df_name, df_i in df.groupby("ID"):
        data_params = model.config_normalization.get_data_params(df_name)
        df_i.drop("ID", axis=1, inplace=True)
        df_aux = df_utils.normalize(df_i, data_params).copy(deep=True)
        df_aux["ID"] = df_name
        df_norm = pd.concat((df_norm, df_aux), ignore_index=True)
    return df_norm
