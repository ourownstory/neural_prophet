.. NeuralProphet documentation master file, created by
    sphinx-quickstart on Tue Oct 12 13:27:59 2021.
    You can adapt this file completely to your liking, but it should at least
    contain the root `toctree` directive.

=========================================
NeuralProphet
=========================================

Based on Neural Networks, inspired by `Facebook Prophet <https://github.com/facebook/prophet>`_ and `AR-Net <https://github.com/ourownstory/AR-Net>`_, built on Pytorch.


Links
-----

- `Read the paper <https://arxiv.org/abs/2111.15397?fbclid=IwAR2vCkHYiy5yuPPjWXpJgAJs-uD5NkH4liORt1ch4a6X_kmpMqagGtXyez4>`_
- `GitHub repository <https://github.com/ourownstory/neural_prophet>`_

Why NeuralProphet?
------------------

NeuralProphet changes the way time series modelling and forecasting is done:

- Support for auto-regression and covariates.
- Automatic selection of training related hyperparameters.
- Fourier term seasonality at different periods such as yearly, daily, weekly, hourly.
- Piecewise linear trend with optional automatic changepoint detection.
- Plotting for forecast components, model coefficients and final predictions.
- Support for global modeling.
- Lagged and future regressors.
- Sparsity of coefficients through regularization.
- User-friendly and powerful Python package:

.. code-block:: pycon

    >>> from neuralprophet import NeuralProphet
    >>> m = NeuralProphet()
    >>> metrics = m.fit(your_df, freq='D')
    >>> forecast = m.predict(your_df)
    >>> m.plot(forecast)


Installing
----------

NeuralProphet can be installed with `pip <https://pypi.org/project/neuralprophet/>`_:

.. code-block:: bash

    $ pip install neuralprophet

If you plan to use the package in a Jupyter notebook, we recommend to install the 'live' version:

.. code-block:: bash

    $ pip install neuralprophet[live]

Alternatively, you can get the most up to date version by cloning directly from `GitHub <https://github.com/ourownstory/neural_prophet>`_:

.. code-block:: bash

    $ git clone https://github.com/ourownstory/neural_prophet.git
    $ cd neural_prophet
    $ pip install .


.. toctree::
    :hidden:
    :maxdepth: 1

    Home<self>
    Quick Start Guide<quickstart>

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Tutorials

    Tutorials<tutorials/tutorials>
    New Tutorials (Preview)<new-tutorials/index>

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: How To Guides

    Guides<guides/index>

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: The Science Behind

    Model Overview<science-behind/model-overview>
    Presentation<https://github.com/ourownstory/neural_prophet/raw/61f1c6d4667db19a189e15037eb230ee5e90b80c/notes/NeuralProphet_Introduction.pdf>

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: API Reference

    forecaster.py <code/forecaster>

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Community

    Contribution<community/contribute>
    GitHub <https://github.com/ourownstory/neural_prophet>
    Slack <https://join.slack.com/t/neuralprophet/shared_invite/zt-1iyfs2pld-vtnegAX4CtYg~6E~V8miXw>
