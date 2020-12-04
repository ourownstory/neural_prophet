import logging

log = logging.getLogger("nprophet")
log.setLevel("INFO")
# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("logs.log", "w+")
# c_handler.setLevel("WARNING")
# f_handler.setLevel("INFO")
# Create formatters and add it to handlers
c_format = logging.Formatter("%(levelname)s: %(name)s - %(funcName)s: %(message)s")
f_format = logging.Formatter("%(asctime)s; %(levelname)s; %(name)s; %(funcName)s; %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
# Add handlers to the logger
log.addHandler(c_handler)
log.addHandler(f_handler)

from .forecaster import NeuralProphet
from .utils import set_random_seed
