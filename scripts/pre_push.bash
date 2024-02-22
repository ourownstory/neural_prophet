#!/usr/bin/env bash

echo "Running pre-push hook: pytest"
if ! pytest tests -v --durations=0;
then
 echo "Failed pytests. Pytests must pass before push!"
 exit 1
fi
