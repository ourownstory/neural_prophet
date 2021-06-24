Feature list for newcomers (each worth a few hours up to a weeks work):

Documentation:
re-structure documentation - with the usage of Sphinx as well
Automate the markdown documentation: create from ipynb
auto-generate documentation from docstrings
Tutorial notebook: how to use in R

Testing:
start using Travis CI - check if it is being done
switch from unittests to pytest 
Extend the coverage of tests (measure coverage and add missing tests) (#339, #340)

Plotting:
Add Plotly - first part done by another collaborator. Check the progress and ad anything missing (compared to the matplotlib functions)

Data Ingestion:
A more forgiving data input mechanism (manage mis-formated datetimes, no datestamps, etc)
fail with specific error message when data contains duplicate date entries
raise error if certain periodic data is missing (e.g. no Sundays) because furier terms would be random there.

Small Issues:
change all uses of AttrDict to dataclasses or named tuples 
add model saving/loading functions
add model.get_params and set_params (for scenario testing) 

Functionality (if you are looking for a challenge):
Conditinal Seasonality (Quarterly, weekday/weekend, daily, user defined)
logistic growth (90% done)
reorganized model blocks to modular functions - modularization: from if-else to (parallel) execution of a list of modules
transition to pytorch lightning
positivity constraint for predictions
multi-targets aka multivariate modelling capability
