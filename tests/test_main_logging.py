import logging

import viam.logging as viam_logging


def test_module_loggers_are_routed_through_viam_logging():
    import vla.main  # noqa: F401 - importing the entrypoint configures logging

    # Registered with viam.logging, so viam-server receives vla.* records at
    # their own level (INFO included), not just WARNING+ via the last-resort handler.
    assert viam_logging.LOGGERS.get("vla") is logging.getLogger("vla")
    assert logging.getLogger("vla").handlers
    assert logging.getLogger("vla.policy.installer").getEffectiveLevel() <= logging.INFO
