"""
Invokes neuralprophet when module is run as a script.
"""
import argparse

from neuralprophet._version import __version__


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="NeuralProphet")
    parser.add_argument("-V", "--version", action="version", version="%(prog)s " + __version__)
    return parser.parse_args(args)


if __name__ == "__main__":
    parse_args()
