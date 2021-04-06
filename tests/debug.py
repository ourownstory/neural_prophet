#!/usr/bin/env python3
import logging
import test_integration
import test_unit
from neuralprophet import NeuralProphet

log = logging.getLogger("nprophet.test.debug")
log.setLevel("INFO")


def debug_logger():
    log.info("testing: Logger")
    log.setLevel("ERROR")
    log.parent.setLevel("WARNING")
    log.warning("### this WARNING should not show ###")
    log.parent.warning("this WARNING should show")
    log.error("this ERROR should show")

    log.setLevel("DEBUG")
    log.parent.setLevel("ERROR")
    log.debug("this DEBUG should show")
    log.parent.warning("### this WARNING not show ###")
    log.error("this ERROR should show")
    log.parent.error("this ERROR should show, too")
    # test existing test cases
    # test_all(log_level="DEBUG")

    # test the set_log_level function
    log.parent.parent.setLevel("INFO")
    m = NeuralProphet(
        n_forecasts=3,
        n_lags=5,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        log_level="DEBUG",
        epochs=5,
    )
    log.parent.parent.debug("this DEBUG should show")
    m.set_log_level(log_level="WARNING")
    log.parent.parent.debug("### this DEBUG should not show ###")
    log.parent.parent.info("### this INFO should not show ###")


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
    itests.test_predict()
    itests.test_plot()
    itests.test_air_data()
    itests.test_random_seed()
    itests.test_loss_func()


def debug_unit_all(plot=False):
    test_unit.UnitTests.plot = plot

    utests = test_unit.UnitTests()
    utests.test_impute_missing()
    utests.test_time_dataset()
    utests.test_normalize()
    utests.test_auto_batch_epoch()
    utests.test_train_speed()


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

    test_integration.IntegrationTests.plot = plot
    itests = test_integration.IntegrationTests()
    ##
    # itests.test_trend()

    test_unit.UnitTests.plot = plot
    utests = test_unit.UnitTests()
    ##
    # utests.test_train_speed()


if __name__ == "__main__":
    # TODO: add argparse to allow for plotting with tests using command line
    # TODO: add hard performance criteria to training tests, setting seeds
    # debug_logger()
    debug_all()
    # debug_one()
