# Contribute
## Dev Install
After downloading the code repository (via `git clone`), change to the repository directory (`cd neural_prophet`), activate your virtual environment, and install neuralprophet as python package with
`pip install -e .[dev]`

(Including the optional `-e` flag will install neuralprophet in "editable" mode, meaning that instead of copying the files into your virtual environment, a symlink will be created to the files where they are.)

Additionally you must run `$ neuralprophet_dev_setup` in your console to run the dev-setup script which installs appropriate git hooks for testing etc.

## Notes
As far as possible, we follow the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html)

As for Git practices, please follow the steps described at [Swiss Cheese](https://github.com/ourownstory/swiss-cheese/blob/master/git_best_practices.md) for how to git-rebase-squash when working on a forked repo.
