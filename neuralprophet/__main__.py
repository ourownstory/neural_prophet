"""
Invokes neuralprophet when module is run as a script.
"""
import argparse
from neuralprophet._version import __version__


parser = argparse.ArgumentParser(description="NeuralProphet")
parser.add_argument("-v", "--version", action="version", version="%(prog)s " + __version__)


if __name__ == "__main__":
    parser.parse_args()
