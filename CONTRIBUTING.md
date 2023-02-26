# Contributing to NeuralProphet
:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

Welcome to the Prophet community and thank you for your contribution to its continued legacy. 
We compiled this page with practical instructions and further resources to help you get started.

Please come join us on our [Slack](https://join.slack.com/t/neuralprophet/shared_invite/zt-sgme2rw3-3dCH3YJ_wgg01IXHoYaeCg), you can message any core dev there.

## Get Started On This
We created an [Contributing: Issues Overview Page](https://github.com/users/ourownstory/projects/3/views/1) with all tasks where we would appreciate your help! 
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
* [Tips for Windows User](https://github.com/ourownstory/neural_prophet/wiki#tips-for-windows-user)

## Writing documentation
The NeuralProphet documentation website is hosted via GitHub Pages on www.neuralprohet.com. Have a look at the [wiki](https://github.com/ourownstory/neural_prophet/wiki#writing-documentation) on how to write and build documentation.

## Best practices
We follow a set of guidelines and methodologies to ensure that code is of high quality, maintainable, and easily understandable by others who may contribute to the project:
* [Typing](https://github.com/ourownstory/neural_prophet/wiki#typing): Use type annotations across the project to improve code readability and maintainability
* [Tests and Code Coverage](https://github.com/ourownstory/neural_prophet/wiki#testing-and-code-coverage): Run tests using 'PyTest' to ensure that the code is functioning as expected.
* [Continuous Integration](https://github.com/ourownstory/neural_prophet/wiki#continous-integration): Github Actions is used to set up a CI pipeline
* [Style](https://github.com/ourownstory/neural_prophet/wiki#style): Deploy Black, so there is no need to worry about style.
* [Pull requests](https://github.com/ourownstory/neural_prophet/wiki#pull-requests) are categorized with a prefix and [labels](https://github.com/ourownstory/neural_prophet/wiki#labels) are assigned to Pull requests and issues to indicate e.g. status or changes.
