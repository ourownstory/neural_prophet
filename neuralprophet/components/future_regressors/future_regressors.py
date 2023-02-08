import logging

from neuralprophet import utils
from neuralprophet.components import BaseComponent

log = logging.getLogger("NP.future_regressors")


class FutureRegressors(BaseComponent):
    def __init__(self, config, id_list, quantiles, n_forecasts, device, config_trend_none_bool):
        super().__init__(n_forecasts=n_forecasts, quantiles=quantiles, id_list=id_list, device=device)

        self.config_regressors = config  # config_regressors
        self.regressors_dims = utils.config_regressors_to_model_dims(config)  # config_regressors
        if self.regressors_dims is not None:
            self.n_additive_regressor_params = 0
            self.n_multiplicative_regressor_params = 0
            for name, configs in self.regressors_dims.items():
                if configs["mode"] not in ["additive", "multiplicative"]:
                    log.error("Regressors mode {} not implemented. Defaulting to 'additive'.".format(configs["mode"]))
                    self.regressors_dims[name]["mode"] = "additive"
                if configs["mode"] == "additive":
                    self.n_additive_regressor_params += 1
                elif configs["mode"] == "multiplicative":
                    if config_trend_none_bool:
                        log.error("Multiplicative regressors require trend.")
                        raise ValueError
                    self.n_multiplicative_regressor_params += 1

        else:
            self.config_regressors = None
