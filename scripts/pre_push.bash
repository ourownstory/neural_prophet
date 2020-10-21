#!/usr/bin/env bash

echo "Running pre-commit hook"
./scripts/run_unittests.bash

# $? stores exit value of the last command
if [ $? -ne 0 ]; then
 echo "Tests must pass before push!"
 exit 1
fi