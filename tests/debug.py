#!/usr/bin/env python3
import logging
import test_integration
import test_unit
from neuralprophet import NeuralProphet, set_log_level

log = logging.getLogger("NP.test.debug")
log.setLevel("INFO")


def test_debug_logger():
    log.info("testing: Logger")
    log.setLevel("ERROR")
    log.parent.setLevel("WARNING")
    log.warning("### this WARNING should not show ###")
    log.parent.warning("1--- this WARNING should show")
    log.error("2--- this ERROR should show")

    log.setLevel("DEBUG")
    log.parent.setLevel("ERROR")
    log.debug("3--- this DEBUG should show")
    log.parent.warning("### this WARNING not show ###")
    log.error("4--- this ERROR should show")
    log.parent.error("5--- this ERROR should show, too")
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
        epochs=5,
    )
    set_log_level("DEBUG")
    log.parent.parent.debug("6--- this DEBUG should show")
    set_log_level(log_level="WARNING")
    log.parent.parent.debug("### this DEBUG should not show ###")
    log.parent.parent.info("### this INFO should not show ###")


def test_debug_integration_all(plot=False):
    test_integration.plot = plot

    itests = test_integration

    itests.test_names()
    itests.test_train_eval_test()
    itests.test_trend()
    itests.test_custom_changepoints()
    itests.test_no_trend()
    itests.test_seasons()
    itests.test_custom_seasons()
    itests.test_ar()
    itests.test_ar_sparse()
    itests.test_ar_deep()
    itests.test_lag_reg()
    itests.test_lag_reg_deep()
    itests.test_events()
    itests.test_future_reg()
    itests.test_plot()
    itests.test_air_data()
    itests.test_random_seed()
    itests.test_loss_func()
    itests.test_yosemite()
    itests.test_model_cv()
    itests.test_callable_loss()


def debug_unit_all(plot=False):
    test_unit.UnitTests.plot = plot

    utests = test_unit.UnitTests()
    #
    utests.test_impute_missing()
    utests.test_time_dataset()
    utests.test_normalize()
    utests.test_auto_batch_epoch()
    utests.test_train_speed_custom()
    utests.test_train_speed_auto()
    utests.test_split_impute()
    utests.test_cv()
    utests.test_reg_delay()


def debug_all():
    pass
    # run from neuralprophet folder:
    # python3 -m unittest discover -s tests


def debug_one(verbose=True):
    if verbose:
        # very verbose option
        plot = True
        log.setLevel("DEBUG")
        log.parent.setLevel("DEBUG")
        log.parent.parent.setLevel("DEBUG")
    else:
        plot = False
        log.setLevel("INFO")
        log.parent.setLevel("INFO")
        log.parent.parent.setLevel("WARNING")

    test_integration.IntegrationTests.plot = plot
    itests = test_integration.IntegrationTests()

    test_unit.UnitTests.plot = plot
    utests = test_unit.UnitTests()
    ##
    utests.test_double_crossvalidation()
    ##
    itests.test_global_modeling()


if __name__ == "__main__":
    # debug_logger()
    # debug_all()
    # debug_one()
    test_debug_integration_all()
