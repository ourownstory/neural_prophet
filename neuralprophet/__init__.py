import logging
import warnings

import pytorch_lightning as pl

# make core features and version number accessible
from ._version import __version__  # noqa: F401
from .df_utils import split_df  # noqa: F401
from .forecaster import NeuralProphet  # noqa: F401
from .torch_prophet import TorchProphet  # noqa: F401
from .utils import load, save, set_log_level, set_random_seed  # noqa: F401

# Reduce lightning logs
warnings.simplefilter(action="ignore", category=pl.utilities.warnings.PossibleUserWarning)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

log = logging.getLogger("NP")
log.setLevel("INFO")

c_handler = logging.StreamHandler()
# c_handler.setLevel("WARNING")
c_format = logging.Formatter("%(levelname)s - (%(name)s.%(funcName)s) - %(message)s")
c_handler.setFormatter(c_format)
log.addHandler(c_handler)

logging.captureWarnings(True)
warnings_log = logging.getLogger("py.warnings")
warnings_log.addHandler(c_handler)

write_log_file = False
if write_log_file:
    f_handler = logging.FileHandler("logs.log", "w+")
    # f_handler.setLevel("ERROR")
    f_format = logging.Formatter("%(asctime)s; %(levelname)s; %(name)s; %(funcName)s; %(message)s")
    f_handler.setFormatter(f_format)
    log.addHandler(f_handler)
    warnings_log.addHandler(f_handler)
