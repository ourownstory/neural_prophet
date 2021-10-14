### 0.2.8 
* Robustify automatic batch_size and epochs selection
* Robustify automatic learning_rate selection based on lr-range-test
* Improve train optimizer and scheduler
* soft-start regularization in last third of training
* Improve reqularization function for all components
* allow custom optimizer and loss_func
* support python 3.6.9 for colab
* Crossvalidation utility
* Chinese documentation
* support callable loss
* Robustify changepoints data format
* require log_level in logger util
* Rename tqdm, remove overbleed option
* Reg schedule: increasing regularization in last third of training
* bug fix in plot country holidays
* Add Energy datasets and example notebook
* disable log file by default
* add double crossvalidation
* improve tests
* Buxfixes

### 0.2.7 
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
