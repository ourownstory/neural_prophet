import logging

# matplotlib is known to spam logs, thus we set the logging level to warning
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.INFO)

# prevent PIL from spaming the logs
logging.getLogger("PIL").setLevel(logging.INFO)
