#!/usr/bin/env python3
import logging
import test_integration
import test_unit

log = logging.getLogger("nprophet.test.debug")
log.setLevel("INFO")


def debug_integration_all(plot=False):
    test_integration.IntegrationTests.plot = plot

    itests = test_integration.IntegrationTests()

    itests.test_names()
    itests.test_train_eval_test()
    itests.test_trend()
    itests.test_no_trend()
    itests.test_seasons()
    itests.test_custom_seasons()
    itests.test_ar_net()
    itests.test_lag_reg()
    itests.test_events()
    itests.test_future_reg()
    itests.test_events()
    itests.test_predict()
    itests.test_plot()
    itests.test_logger()


def debug_unit_all(plot=False):
    test_unit.UnitTests.plot = plot

    utests = test_unit.UnitTests()
    utests.test_impute_missing()
    utests.test_time_dataset()


def debug_integration(plot=False):
    test_integration.IntegrationTests.plot = plot
    itests = test_integration.IntegrationTests()

    # to run individual tests
    # itests.test_names()
    # itests.test_train_eval_test()
    itests.test_trend()
    # itests.test_no_trend()
    # itests.test_seasons()
    # itests.test_custom_seasons()
    # itests.test_ar_net()
    # itests.test_lag_reg()
    # itests.test_events()
    # itests.test_future_reg()
    # itests.test_events()
    # itests.test_predict()
    # itests.test_plot()
    # itests.test_logger()


def debug_unit(plot=False):
    test_unit.UnitTests.plot = plot
    utests = test_unit.UnitTests()

    # to run individual tests
    # utests.test_impute_missing()
    # utests.test_time_dataset()


def debug_all():
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
    debug_unit_all(plot)
    debug_integration_all(plot)


def debug_one():
    # default
    # plot = False
    # log.setLevel("INFO")
    # log.parent.setLevel("DEBUG")
    # log.parent.parent.setLevel("WARNING")

    # very verbose option
    plot = True
    log.setLevel("DEBUG")
    log.parent.setLevel("DEBUG")
    log.parent.parent.setLevel("DEBUG")

    debug_unit(plot)
    debug_integration(plot)


if __name__ == "__main__":
    # TODO: add argparse to allow for plotting with tests using command line
    # TODO: add hard performance criteria to training tests, setting seeds
    # debug_all()
    debug_one()
