#!/usr/bin/env bash

echo "Running pre-push hook: pytest"
if ! coverage run -m pytest -v;
then
 echo "Failed pytests. Pytests must pass before push!"
 exit 1
fi