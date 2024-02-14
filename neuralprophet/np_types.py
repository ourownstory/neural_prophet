import sys
from typing import Dict, Union

import torch

# Ensure compatibility with python 3.7
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


NormalizeMode = Literal["auto", "soft", "soft1", "minmax", "standardize", "off"]

SeasonalityMode = Literal["additive", "multiplicative"]

SeasonalityArgument = Union[Literal["auto"], bool, int]

GrowthMode = Literal["off", "linear", "discontinuous"]

CollectMetricsMode = Union[Dict, bool]

SeasonGlobalLocalMode = Literal["global", "local", "glocal"]

FutureRegressorsModel = Literal["linear", "neural_nets", "shared_neural_nets"]

Components = Dict[str, torch.Tensor]
