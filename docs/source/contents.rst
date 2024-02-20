.. NeuralProphet documentation master file, created by
    sphinx-quickstart on Tue Oct 12 13:27:59 2021.
    You can adapt this file completely to your liking, but it should at least
    contain the root `toctree` directive.

=========================================
NeuralProphet
=========================================

Fusing traditional time series algorithms using standard deep learning methods, built on PyTorch, inspired by `Facebook Prophet <https://github.com/facebook/prophet>`_ and `AR-Net <https://github.com/ourownstory/AR-Net>`_.


Simple Example
------------------

.. code-block:: pycon

    >>> from neuralprophet import NeuralProphet
    >>> m = NeuralProphet()
    >>> metrics = m.fit(df)
    >>> forecast = m.predict(df)
    >>> m.plot(forecast)

Features
------------------

NeuralProphet provides many time series modeling and workflow features, in a simple package:

- Support for global modeling of many time series.
- Automatic selection of training related hyperparameters.
- Plotting utilities for forecast components, model coefficients and final predictions.
- Local context through Autoregression and lagged covariates.
- Changing trends and smooth seasonality at different periods.
- Modeling of event, holiday, and future regressor effects.
- Many customization options, such as regularization.

Resources
-----------------

- `Read the paper <https://arxiv.org/abs/2111.15397?fbclid=IwAR2vCkHYiy5yuPPjWXpJgAJs-uD5NkH4liORt1ch4a6X_kmpMqagGtXyez4>`_
- `GitHub repository <https://github.com/ourownstory/neural_prophet>`_


.. toctree::
    :hidden:
    :maxdepth: 1

    Home<self>
    Quick Start Guide<quickstart>

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Tutorials

    Tutorials<tutorials/index>

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: How To Guides

    Guides<how-to-guides/index>

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Code Documentation

    NeuralProphet <code/index>

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: About

    Model Overview<science-behind/model-overview>
    Presentation<https://github.com/ourownstory/neural_prophet/raw/61f1c6d4667db19a189e15037eb230ee5e90b80c/notes/NeuralProphet_Introduction.pdf>

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Community

    Contribute<community/contribute>
    GitHub<https://github.com/ourownstory/neural_prophet>
    Slack<https://join.slack.com/t/neuralprophet/shared_invite/zt-1iyfs2pld-vtnegAX4CtYg~6E~V8miXw>

