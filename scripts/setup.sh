#!/bin/sh

echo "Setting up the environment..."
python3 -m venv .venv

echo "Activating the environment..."
source .venv/bin/activate

echo "Installing dependencies..."
pip install -U pip
pip install -e .[dev]
