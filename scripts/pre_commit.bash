#!/bin/sh

set -e

files=$(git diff --staged --name-only --diff-filter=d -- "*.py")
for file in $files; do
  black "$file"
  git add "$file"
done