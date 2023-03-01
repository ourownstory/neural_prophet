# Contributing to NeuralProphet
:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

Welcome to the Prophet community and thank you for your contribution to its continued legacy. 
We compiled this page with practical instructions and further resources to help you get started.

For an easy start, check out all open issues with the label https://github.com/ourownstory/neural_prophet/labels/good%20first%20issue. 
They can be done somewhat in isolation from other tasks and will take a couple hours up to a week of work to complete. We appreciate your help!

Please come join us on our [Slack](https://join.slack.com/t/neuralprophet/shared_invite/zt-sgme2rw3-3dCH3YJ_wgg01IXHoYaeCg), you can message any core dev there.

## Process
In summary, follow the steps below to make your first contribution. If you want to have more details on every step, check out this [beginner's guide to contributing to a GitHub project](https://akrabat.com/the-beginners-guide-to-contributing-to-a-github-project/#to-sum-up). 

1. Setup your development environment for NeuralProphet following [this tutorial](https://github.com/ourownstory/neural_prophet/edit/1166-shorten-contributing.md/CONTRIBUTING.md#dev-install) 
2. Find an issue to work on by filtering on the label https://github.com/ourownstory/neural_prophet/labels/good%20first%20issue
3. Pull the latest changes from remote: `git pull upstream main`
4. Push changes to your fork: `git push origin main`
5. Create a new branch from main to work on: `git checkout -b BRANCH_NAME`
6. Make some changes in your local repository 
7. Stage the changes: `git add -A`
8. Commit the changes: `git commit -m "DESCRIPTION OF CHANGES"`
9. Do a pull request: If you visit https://github.com/ourownstory/neural_prophet now, there should be a green button saying "Compare & pull request", click on it and describe the changes you did. Now you only need to wait for a review and respond to any feedback.

Congratulations, you just contributed your first issue. For your next issue just follow the steps starting from number 2.

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
neuralprophet_dev_setup.py
git config pull.ff only 
```

Notes: 
* Including the optional `-e` flag will install neuralprophet in "editable" mode, meaning that instead of copying the files into your virtual environment, a symlink will be created to the files where they are.
* The `neuralprophet_dev_setup` command runs the dev-setup script which installs appropriate git hooks for Black (pre-commit) and PyTest (pre-push).
* setting git to fast-forward only prevents accidental merges when using `git pull`.
* To run tests without pushing (or when the hook installation fails), run from neuralprophet folder: `pytest -v`
* To run black without commiting (or when the hook installation fails): `python3 -m black {source_file_or_directory}` 
* If running `neuralprophet_dev_setup.py` gives you a `no such file` error, try running `python ./scripts/neuralprophet_dev_setup.py`

## Writing documentation
The NeuralProphet documentation website is hosted via GitHub Pages on www.neuralprohet.com. Have a look at the [wiki](https://github.com/ourownstory/neural_prophet/wiki#writing-documentation) on how to write and build documentation.

## Best practices
We follow a set of guidelines and methodologies to ensure that code is of high quality, maintainable, and easily understandable by others who may contribute to the project:
* [Typing](https://github.com/ourownstory/neural_prophet/wiki#typing): Use type annotations across the project to improve code readability and maintainability
* [Tests and Code Coverage](https://github.com/ourownstory/neural_prophet/wiki#testing-and-code-coverage): Run tests using 'PyTest' to ensure that the code is functioning as expected.
* [Continuous Integration](https://github.com/ourownstory/neural_prophet/wiki#continous-integration): Github Actions is used to set up a CI pipeline
* [Code Style](https://github.com/ourownstory/neural_prophet/wiki#style): Deploy Black, so there is no need to worry about code style and formatting.

# Github issues and pull requests

## Pull requests

### Prefixes for pull requests

All pull requests should have one of the following prefixes:
* [Breaking] Breaking changes, which require user action (e.g. breaking API changes)
* [Major] Major features worth mentioning (e.g. uncertainty prediction)
* [Minor] Minor features which are nice to know about (e.g. add sorting to labels in plots)
* [Fix] Bugfixes (e.g. fix for plots not showing up)
* [Docs] Documentation related changes (e.g. add tutorial for energy dataset)
* [Tests] Tests additions and changes (e.g. add tests for utils)
* [DevOps] Github workflows (e.g. add pyright type checking Github action)

Those prefixed are then used to generate the changelog and decide which version number change is necessary for a release.

### Labels for pull requests

Once the development of a new feature is 
- https://github.com/ourownstory/neural_prophet/labels/status%3A%20blocked
- https://github.com/ourownstory/neural_prophet/labels/status%3A%20needs%20review
- https://github.com/ourownstory/neural_prophet/labels/status%3A%20needs%20update
- https://github.com/ourownstory/neural_prophet/labels/status%3A%20ready

## Issue labels

Issues should always have a type and a priority. Other labels are optional.

**Issue type**

https://github.com/ourownstory/neural_prophet/labels/bug
https://github.com/ourownstory/neural_prophet/labels/epic
https://github.com/ourownstory/neural_prophet/labels/task
(questions should be moved to [discussions](https://github.com/ourownstory/neural_prophet/discussions))

**Priorities**

https://github.com/ourownstory/neural_prophet/labels/P1
https://github.com/ourownstory/neural_prophet/labels/P2
https://github.com/ourownstory/neural_prophet/labels/P3

**Getting started**

https://github.com/ourownstory/neural_prophet/labels/good%20first%20issue

**Closed for reason**

https://github.com/ourownstory/neural_prophet/labels/duplicate
https://github.com/ourownstory/neural_prophet/labels/wontfix
