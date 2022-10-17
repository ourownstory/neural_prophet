# make version number accessible
from ._version import __version__  # noqa: F401

# make core features accessible
from .forecaster import NeuralProphet  # noqa: F401
from .utils import set_random_seed, set_log_level  # noqa: F401
from .utils import save, load  # noqa: F401
from .df_utils import split_df  # noqa: F401


import logging

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
