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

## Writing documentation
NeuralProphet uses the Sphinx documentation framework to build the documentation website, which is hosted via Github Pages on [www.neuralprophet.com](http://www.neuralprophet.com).

The documentation's source is enclosed in the docs folder. Whereas the `main` branch only contains the basic source files, the branch `gh-pages` entails the build data (with folders `docs/html` and `docs/doctrees`) and is used for deployment.

### Docstring

Docstrings need to be formatted according to [NumPy Style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy) in order to display their API reference correctly using Spinx. 
Please refer to [Pandas Docstring Guide](https://pandas.pydata.org/pandas-docs/stable/development/contributing_docstring.html#) for best practices.

The length of line inside docstrings block must be limited to 80 characters to fit into Jupyter documentation popups.

You can check for adherence to the style guide by running:
```sh
pydocstyle --convention=numpy path/my_file.py
```
(You may need to install the tool first. On Linux: `sudo apt install pydocstyle`.)


#### Example 
See how Pandas does this for `melt` in their [melt documentation page](https://pandas.pydata.org/docs/reference/api/pandas.melt.html) and how it looks in the [melt docstring](https://github.com/pandas-dev/pandas/blob/v1.4.1/pandas/core/shared_docs.py#L153).

Docstring architecture sample:

```
def return_first_elements(n=5):
    """
    Return the first elements of a given Series.

    This function is mainly useful to preview the values of the
    Series without displaying all of it.

    Parameters
    ----------
    n : int
        Number of values to return.

    Return
    ------
    pandas.Series
        Subset of the original series with the n first values.

    See Also
    --------
    tail : Return the last n elements of the Series.
    Examples
    --------
    If you have multi-index columns:
    >>> df.columns = [list('ABC'), list('DEF')]
    >>> df
       A  B  C
       D  E  F
    0  a  1  2
    1  b  3  4
    2  c  5  6
    """
    return self.iloc[:n]
```

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

3. Create PR to merge new branch into main.

4. After merge: Checkout `gh-pages`, navigate to `cd docs\` and generate the documentation HTML files. The generated files will be in `docs/build/html`.

```bash
make html
```

Notes:
* If you get an error that involves `Pandoc not found` - install pandoc manually on your operating system. For linux: `sudo apt install pandoc`

5. Commit and push changes to branch `gh-pages`. Changes should be reflected instantly on the [documentation website](http://www.neuralprophet.com).

## Typing

We try to use type annotations across the project to improve code readability and maintainability.

Please follow the official python recommendations for [type hints](https://docs.python.org/3/library/typing.html) and [PEP-484](https://peps.python.org/pep-0484/).

### Postponing the evaluation type annotations and python version

The Postponed Evaluation of Annotations [PEP 563](https://docs.python.org/3/whatsnew/3.7.html#pep-563-postponed-evaluation-of-annotations) provides major benefits for type annotations. To use them with our currently support python versions we must use the following syntax:

```python
from __future__ import annotations
```

### Circular imports with type annotations

When using type annotations, you may encounter circular imports. To avoid this, you can use the following pattern based on the [typing.TYPE_CHECKING](https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING) constant:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

# Imports only needed for type checking
if TYPE_CHECKING:
    from my_module import MyType
```

## Testing and Code Coverage

We are using `PyTest` to run tests within our projects. All tests can be found in `tests/` directory. 

All tests can be triggered via the command: 

```bash
pytest tests -v
```

Running specific tests can be done by running the command: 

```bash
pytest tests -k "name_of_test"
```

We are using [pytest-cov](https://pypi.org/project/pytest-cov/) and [codecov](https://app.codecov.io/gh/ourownstory/neural_prophet) to create transparent code coverage reports.
To locally trigger and output a code coverage report via the commandline, run the following command: 

```bash
pytest tests -v --cov=./
```


## Continous Integration

We are using Github Actions to setup a CI pipeline. The creation as well as single commits to a pull request trigger the CI pipeline.

Currently there is one workflow called `.github/worklfows/ci.yml` to trigger testing, create code coverage reports via [pytest-cov](https://pypi.org/project/pytest-cov/) and subsequently uploading reports via [codecov](https://app.codecov.io/gh/ourownstory/neural_prophet) for the major OS systems (Linux, Mac, Windows). 


## Style
We deploy Black, the uncompromising code formatter, so there is no need to worry about style. Beyond that, where reasonable, for example for docstrings, we follow the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html)

As for Git practices, please follow the steps described at [Swiss Cheese](https://github.com/ourownstory/swiss-cheese/blob/master/git_best_practices.md) for how to git-rebase-squash when working on a forked repo. (Update: all PR are now squashed, so you can skip this step, but it's still good to know.)

### String formatting
Please use the more readable [f-string formatting style](https://docs.python.org/3/tutorial/inputoutput.html).

## Tips for Windows User:
To contribute to NeuralProphet from Windows install WSL to run Linux terminal in Windows.

1.Install WSL2.

2.Install libraries 

   a. pip:This will allow users to quick install using pip.
   
```bash
sudo apt install pip
```
    
   b.For any ”name” not found try.
   
```bash
pip install <name>
```
Notes: 
- To install NeuralProphet in dev mode, create a venv using the Linux terminal on the subsystem drive (not the mount).
- For any statement error try using sudo and --user which will then allow administrator access to perform the action.


