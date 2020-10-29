The documentation is a work in progress.

# Quick Start 
## Install
After downloading the code repository (via `git clone`), change to the repository directory (`cd neural_prophet`) and install neuralprophet as python package with
`pip install .`

Note: If you plan to use the package in a Jupyter notebook, it is recommended to install the 'live' package version with `pip install .[live]`.
This will allow you to enable `plot_live_loss` in the `train` function to get a live plot of train (and validation) loss.

Now you can use NeuralProphet in your code:
```python
from neuralprophet import NeuralProphet
model = NeuralProphet()
```