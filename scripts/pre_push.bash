#!/usr/bin/env bash

echo "Running pre-push hook: unittests"
if ! python3 -m unittest discover -s tests;
then
 echo "Failed tests. Unittests must pass before push!"
 exit 1
fi