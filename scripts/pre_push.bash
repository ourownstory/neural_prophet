#!/usr/bin/env bash

echo "Running pre-push hook"

python3 -m unittest discover -s tests

# $? stores exit value of the last command
if [ $? -ne 0 ]; then
 echo "Tests must pass before push!"
 exit 1
fi