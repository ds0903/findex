"""Налаштування логування. Викликається один раз, у точці входу CLI, не в бібліотечному коді."""

import logging
import sys

_HANDLER_NAME = "findex-cli"


def level_for(verbose: int) -> int:
    """0 -> WARNING, -v -> INFO, -vv (і більше) -> DEBUG."""
    if verbose <= 0:
        return logging.WARNING
    return logging.INFO if verbose == 1 else logging.DEBUG


def setup_logging(verbose: int = 0) -> None:
    """Усі повідомлення йдуть у stderr, щоб не псувати дані в stdout (наприклад, --json)."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() == _HANDLER_NAME:  # свій, від попереднього виклику
            root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(level_for(verbose))
