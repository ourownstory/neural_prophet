import sys
from typing import Dict, List, Union

import torchmetrics

# Ensure compatibility with python 3.7
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


NormalizeMode = Literal["auto", "soft", "soft1", "minmax", "standardize", "off"]

SeasonalityMode = Literal["additive", "multiplicative"]

SeasonalityArgument = Union[Literal["auto"], bool, int]

GrowthMode = Literal["off", "linear", "discontinuous"]

CollectMetricsMode = Union[List[str], bool, Dict[str, torchmetrics.Metric]]

SeasonGlobalLocalMode = Literal["global", "local"]
