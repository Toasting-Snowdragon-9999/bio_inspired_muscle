"""@brief Shared logger configuration: extends sys.path so the package is importable and
exposes a single pre-configured module-level Logger instance for the project."""

import os
import sys

# Append (not insert-at-0) so we don't shadow callers that already have
# their own packages on sys.path — `sys.path.insert(0, ...)` here previously
# masked the mujoco_gui frontend's local ``sim/`` package. Guarded so we
# don't keep growing sys.path on repeated imports.
_BIM_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _BIM_PYTHON not in sys.path:
    sys.path.append(_BIM_PYTHON)

from logger.logger import Logger, LogLvl

logger = Logger(log_level=LogLvl.DEBUG, log_file="test.log", log_to_file=True)


if __name__ == "__main__":
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.error("This is an error message.")
    logger.console("This is a console message.")