# Contributing to NeuralProphet
:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

Welcome to the Prophet community and thank you for your contribution to its continued legacy. 
We compiled this page with practical instructions and further resources to help you get started.

Please come join us on our [Slack](https://join.slack.com/t/neuralprophet/shared_invite/zt-sgme2rw3-3dCH3YJ_wgg01IXHoYaeCg), you can message any core dev there.

## Get Started On This
We created a [overview page](https://github.com/ourownstory/neural_prophet/projects/8) with all tasks where we would appreciate your help! 
They can be done somewhat in isolation from other tasks and will take a couple hours up to a week of work to complete.

## Process
Here's a great [beginner's guide to contributing to a GitHub project](https://akrabat.com/the-beginners-guide-to-contributing-to-a-github-project/#to-sum-up). 

In Summary: 
* Fork the project & clone locally.
* Create an upstream remote and sync your local copy before you branch.
* Branch for each separate piece of work.
* Do the work, write good commit messages, and read the CONTRIBUTING file if there is one.
* Push to your origin repository.
* Create a new PR in GitHub.
* Respond to any code review feedback.

Please make sure to include tests and documentation with your code.

## Dev Install
Before starting it's a good idea to first create and activate a new virtual environment:
```
python3 -m venv <path-to-new-env>
source <path-to-new-env>/bin/activate
```
Now you can install neuralprophet:

```
git clone <copied link from github>
cd neural_prophet
pip install -e ".[dev]"
```

Please don't forget to run the dev setup script to install the hooks for black and pytest, and set git to fast forward only:
```
neuralprophet_dev_setup
git config pull.ff only 
```

Notes: 
* Including the optional `-e` flag will install neuralprophet in "editable" mode, meaning that instead of copying the files into your virtual environment, a symlink will be created to the files where they are.
* The `neuralprophet_dev_setup` command runs the dev-setup script which installs appropriate git hooks for Black (pre-commit) and PyTest (pre-push).
* setting git to fast-forward only prevents accidental merges when using `git pull`.
* To run tests without pushing (or when the hook installation fails), run from neuralprophet folder: `pytest -v`
* To run black without commiting (or when the hook installation fails): `python -m black {source_file_or_directory}` 

## Writing documentation
NeuralProphet's documentation website is hosted via Github Pages on [www.neuralprophet.com](http://www.neuralprophet.com).

NeuralProphet uses the Sphinx documentation framework and [Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for formatting docstrings. 
Length of line inside docstrings block must be limited to 80 characters to fit into Jupyter documentation popups.

The documentation's source is enclosed in the docs folder. Whereas the `master` branch does only contain the basic source files, the branch `gh-pages` entails the build data and is used for deployment.


### Tutorials: Editing existing and adding new
The Jupyter notebooks located inside `tutorials/` are rendered using the Sphinx `nblink` package. 

When you add a new tutorial notebook, please add the tutorial file to the respective section inside `docs/source/contents.rst`.

Next, automatically generate the corresponding `.nblink` files by running this command: 

```bash
python3 docs/check_nblink_files.py
```
In case you changed the name of an existing tutorial please follow the same steps outlined above.

### Building documentation
To build the documentation:

1. Build and install NeuralProphet as described [above](#dev-install).

2. Create a new branch and perform respective documentation changes. 

3. Create PR to merge new branch into master.

4. After merge: Checkout `gh-pages`, navigate to `cd docs\` and generate the documentation HTML files. The generated files will be in `docs/build/html`.

```bash
make html
```

5. Commit and push changes to branch `gh-pages`. Changes should be reflected instantly on the [documentation website](http://www.neuralprophet.com).

## Testing and Code Coverage

We are using `PyTest` to run tests within our projects. All tests can be found in `tests/` directory. 

All tests can be triggered via the command: 

```bash
pytest -v
```

Running specific tests can be done by running the command: 

```bash
pytest tests/ -k "name_of_test"
```

We are using [pytest-cov](https://pypi.org/project/pytest-cov/) and [codecov](https://app.codecov.io/gh/ourownstory/neural_prophet) to create transparent code coverage reports.
To locally trigger and output a code coverage report via the commandline, run the following command: 

```bash
pytest --cov=./
```


## Continous Integration

We are using Github Actions to setup a CI pipeline. The creation as well as single commits to a pull request trigger the CI pipeline.

Currently there is one workflow called `.github/worklfows/ci.yml` to trigger testing, create code coverage reports via [pytest-cov](https://pypi.org/project/pytest-cov/) and subsequently uploading reports via [codecov](https://app.codecov.io/gh/ourownstory/neural_prophet) for the major OS systems (Linux, Mac, Windows). 


## Style
We deploy Black, the uncompromising code formatter, so there is no need to worry about style. Beyond that, where reasonable, for example for docstrings, we follow the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html)

As for Git practices, please follow the steps described at [Swiss Cheese](https://github.com/ourownstory/swiss-cheese/blob/master/git_best_practices.md) for how to git-rebase-squash when working on a forked repo. (Update: all PR are now squashed, so you can skip this step, but it's still good to know.)
