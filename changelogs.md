### 0.2.9 (future)
* confidence interval for forecast (as quantiles via pinball loss)
* Logistic growth for trend component.
* better documentation

### 0.2.8 (upcoming)
* Robustify automatic batch_size and epochs selection
* Robustify automatic learning_rate selection based on lr-range-test
* Improve train optimizer and scheduler
* soft-start regularization in last third of training
* Improve reqularization function for all components
* allow custom optimizer and loss_func
* support python 3.6.9 for colab
* Crossvalidation utility
* Chinese documentation
* bugfixes and UI improvements

### 0.2.7 (current)
* example notebooks: Sub-daily data, Autoregresseion
* bugfixes: `lambda_delay`, `train_speed`

### 0.2.6 
* Auto-set `batch_size` and `epochs`
* add `train_speed` setting
* add `set_random_seed` util
* continued removal of `AttrDict` uses
* bugfix to index issue in `make_future_dataframe`

### 0.2.5
* documentation pages added
* 1cycle policy
* learning rate range test
* tutorial notebooks: trend, events
* fixes to plotting, changepoints
