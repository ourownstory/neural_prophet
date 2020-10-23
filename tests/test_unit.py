#!/usr/bin/env python3

import unittest
import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import logging
from neuralprophet import df_utils

from neuralprophet import NeuralProphet

log = logging.getLogger("nprophet.test_debug")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "example_data")
PEYTON_FILE = os.path.join(DATA_DIR, "wp_log_peyton_manning.csv")


class UnitTests(unittest.TestCase):
    plot = False
    log.setLevel("WARNING")
    log.parent.setLevel("WARNING")

    def test_impute_missing(self,):
        """Debugging data preprocessing"""
        log.info("testing: Impute Missing")
        allow_missing_dates = False

        df = pd.read_csv(PEYTON_FILE)
        name = 'test'
        df[name] = df['y'].values

        if not allow_missing_dates:
            df_na, _ = df_utils.add_missing_dates_nan(df.copy(deep=True), freq='D')
        else:
            df_na = df.copy(deep=True)
        to_fill = pd.isna(df_na['y'])
        # TODO fix debugging printout error
        # log.debug("sum(to_fill)", sum(to_fill.values))

        # df_filled, remaining_na = df_utils.fill_small_linear_large_trend(
        #     df.copy(deep=True),
        #     column=name,
        #     allow_missing_dates=allow_missing_dates
        # )
        df_filled, remaining_na = df_utils.fill_linear_then_rolling_avg(
            df.copy(deep=True),
            column=name,
            allow_missing_dates=allow_missing_dates
        )
        # TODO fix debugging printout error
        # log.debug("sum(pd.isna(df_filled[name]))", sum(pd.isna(df_filled[name]).values))

        if self.plot:
            if not allow_missing_dates:
                df, _ = df_utils.add_missing_dates_nan(df)
            df = df.loc[200:250]
            fig1 = plt.plot(df['ds'], df[name], 'b-')
            fig1 = plt.plot(df['ds'], df[name], 'b.')

            df_filled = df_filled.loc[200:250]
            # fig3 = plt.plot(df_filled['ds'], df_filled[name], 'kx')
            fig4 = plt.plot(df_filled['ds'][to_fill], df_filled[name][to_fill], 'kx')
            plt.show()


if __name__ == '__main__':
    # if called directly
    # TODO: add argparse to allow for plotting with tests using command line
    # TODO: add hard performance criteria to training tests, setting seeds

    # uncomment to run tests with plotting or debug logs print output and  respectively

    # default option
    UnitTests.plot = False
    log.setLevel("DEBUG")
    log.parent.setLevel("WARNING")

    # not verbose option
    # UnitTests.plot = False
    # log.setLevel("ERROR")
    # log.parent.setLevel("ERROR")

    # very verbose option
    # UnitTests.plot = True
    # log.setLevel("DEBUG")
    # log.parent.setLevel("DEBUG")

    tests = UnitTests()

    # to run all tests
    unittest.main(exit=False)

    # to run individual tests
    # tests.test_impute_missing()

