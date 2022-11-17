#!/bin/sh

set -e

pyfiles=$(git diff --staged --name-only --diff-filter=d -- "*.py")
for file in $pyfiles; do
  black "$file"
  isort "$file"
  git add "$file"
done

notebooks=$(git diff --staged --name-only --diff-filter=d -- "*.ipynb")
for file in $notebooks; do
  black "$file"
  git add "$file"
done
