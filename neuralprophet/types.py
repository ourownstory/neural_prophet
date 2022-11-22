import sys

# Ensure compatibility with python 3.7
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


NormalizeMode = Literal["auto", "soft", "soft1", "minmax", "standardize", "off"]

NLagsMode = Literal["auto", "scalar"]
