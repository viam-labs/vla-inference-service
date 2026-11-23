"""Module entrypoint.

Importing the two service classes is what registers them: EasyResource
registers each subclass at class-definition time. run_from_registry then
serves everything in the registry.
"""

import asyncio

from viam import logging as viam_logging
from viam.module.module import Module

from vla.controller.service import VLAController  # noqa: F401 - import registers the model
from vla.policy.service import VLAPolicy  # noqa: F401 - import registers the model

# The vla.* loggers are plain stdlib loggers with no handler, so only WARNING and
# above escaped (via Python's last-resort stderr handler, which viam-server logs
# as errors). Registering their parent with viam.logging forwards every vla.*
# record to viam-server at its own level, honoring the module's configured log
# level -- including INFO lines like the lazy install's progress and "policy ready".
viam_logging.getLogger("vla")

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
