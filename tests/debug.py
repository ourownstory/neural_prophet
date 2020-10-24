#!/usr/bin/env python3
import logging

import test_integration
import test_unit

log = logging.getLogger("nprophet.test.debug")
log.setLevel("INFO")


def debug_integration(plot=False):
    test_integration.IntegrationTests.plot = plot

    tests = test_integration.IntegrationTests()
    # to run individual tests
    tests.test_names()
    tests.test_train_eval_test()
    tests.test_trend()
    tests.test_seasons()
    tests.test_ar_net()
    tests.test_lag_reg()
    tests.test_events()
    tests.test_future_reg()
    tests.test_events()
    tests.test_predict()
    tests.test_plot()
    tests.test_logger()


def debug_unit(plot=False):
    test_unit.UnitTests.plot = plot

    tests = test_unit.UnitTests()
    # to run individual tests
    tests.test_impute_missing()
    tests.test_time_dataset()


if __name__ == '__main__':
    # TODO: add argparse to allow for plotting with tests using command line
    # TODO: add hard performance criteria to training tests, setting seeds
    # default
    plot = False
    log.setLevel("INFO")
    log.parent.setLevel("DEBUG")
    log.parent.parent.setLevel("WARNING")

    # not verbose option
    # plot = False
    # log.setLevel("ERROR")
    # log.parent.setLevel("ERROR")
    # log.parent.parent.setLevel("ERROR")

    # very verbose option
    # plot = True
    # log.setLevel("DEBUG")
    # log.parent.setLevel("DEBUG")
    # log.parent.parent.setLevel("DEBUG")

    debug_unit(plot)
    debug_integration(plot)





