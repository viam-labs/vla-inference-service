"""Module entrypoint.

Importing the two service classes is what registers them: EasyResource
registers each subclass at class-definition time. run_from_registry then
serves everything in the registry.
"""

import asyncio

from viam.module.module import Module

from vla.controller.service import VLAController  # noqa: F401 - import registers the model
from vla.policy.service import VLAPolicy  # noqa: F401 - import registers the model

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
