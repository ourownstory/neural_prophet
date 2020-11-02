#!/usr/bin/env bash

echo "Installing hooks..."
GIT_DIR=$(git rev-parse --git-dir)
# create symlink to our pre-commit and pre-push scripts
ln -s ../../scripts/pre_commit.bash "$GIT_DIR"/hooks/pre-commit
ln -s ../../scripts/pre_push.bash "$GIT_DIR"/hooks/pre-push
# make the symlinks executable
chmod a+rwx "$GIT_DIR"/hooks/pre-commit
chmod a+rwx "$GIT_DIR"/hooks/pre-push
echo "Done!"
