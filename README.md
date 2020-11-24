[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NeuralProphet
A Neural Network based Time-Series model, inspired by [Facebook Prophet](https://github.com/facebook/prophet) and [AR-Net](https://github.com/ourownstory/AR-Net).

For a visual introduction to NeuralProphet, view the [presentation given at the 40th International Symposium on Forecasting (Oct 26, 2020)](notes/Presented_at_International_Symposium_on_Forecasting.pdf).

## Documentation
A proper [documentation page](https://ourownstory.github.io/neural_prophet/) is in the works.

## User Install
After downloading the code repository (via `git clone`), change to the repository directory (`cd neural_prophet`) and install neuralprophet as python package with
`pip install .`

Note: If you plan to use the package in a Jupyter notebook, it is recommended to install the 'live' package version with `pip install .[live]`.
This will allow you to enable `plot_live_loss` in the `train` function to get a live plot of train (and validation) loss.

Now you can use NeuralProphet in your code:
```python
from neuralprophet import NeuralProphet
model = NeuralProphet()
```

## Contribute
### Dev Install
After downloading the code repository (via `git clone`), change to the repository directory (`cd neural_prophet`), activate your virtual environment, and install neuralprophet as python package with
`pip install -e .[dev]`

(Including the optional `-e` flag will install neuralprophet in "editable" mode, meaning that instead of copying the files into your virtual environment, a symlink will be created to the files where they are.)

Additionally you must run `$ neuralprophet_dev_setup` in your console to run the dev-setup script which installs appropriate git hooks for testing etc.

### Notes
We deploy Black, the uncompromising code formatter, so there is no need to worry about style. Beyond that, where reasonable, for example for doicstrings, we follow the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html)

As for Git practices, please follow the steps described at [Swiss Cheese](https://github.com/ourownstory/swiss-cheese/blob/master/git_best_practices.md) for how to git-rebase-squash when working on a forked repo.

## Version History

## Coming up Next


## Development Timeline
For details, please view the [Development Timeline](notes/development_timeline.md).

## Authors
The alpha-stage NeuralProphet was developed by Oskar Triebe, advised by Ram Rajagopal (Stanford University) and Nikolay Laptev (Facebook, Inc), and was funded by Total S.A.
We are now further developing the beta-stage package in collaboration with Hansika Hewamalage, who is advised by Christoph Bergmeir (Monash University).
If you are interested in joining the project, please feel free to reach out to me (Oskar) - you can find my email on the [AR-Net Paper](https://arxiv.org/pdf/1911.12436.pdf).
