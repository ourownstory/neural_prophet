#!/usr/bin/env bash

echo "Running pre-push hook: pytest"
if ! pytest -n auto --durations=0 -c tests/pytest.ini -v tests;
then
 echo "Failed pytests. Pytests must pass before push!"
 exit 1
fi
