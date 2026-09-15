# Utils

```{eval-rst}
.. module:: opendvp.utils
.. currentmodule:: opendvp

.. autosummary::
    :toctree: generated

    utils.get_datetime
    utils.parse_color_for_qupath
```

## Logging

`opendvp.utils.logger` is a pre-configured [loguru](https://loguru.readthedocs.io/) logger,
writing to `stdout` at `INFO` level. Every openDVP function logs its steps through it, so
adjusting it changes openDVP's output:

```python
import sys
from opendvp.utils import logger

logger.remove()  # drop the default openDVP handler
logger.add(sys.stderr, level="DEBUG")
```
