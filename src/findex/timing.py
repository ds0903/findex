import functools
import logging
import time
from collections.abc import Callable


def timed[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Пише в лог (DEBUG), скільки мілісекунд працювала функція.

    functools.wraps копіює назву й docstring: без нього в логах і в help усі
    задекоровані функції називалися б "wrapper".
    """
    log = logging.getLogger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.debug("%s: %.2f мс", func.__qualname__, elapsed_ms)

    return wrapper
