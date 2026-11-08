import time
from dataclasses import dataclass, field


@dataclass
class CrawlStats:
    """Живі лічильники краулера. Пише краулер, читає екран (rich) і підсумок."""

    pages: int = 0  # сторінок віддано споживачу
    errors: int = 0  # URL, які не вдалося завантажити (після повторів)
    retries: int = 0  # скільки разів довелося повторити запит
    in_flight: int = 0  # запитів, що зараз чекають відповіді
    queued: int = 0  # довжина черги адрес
    robots_blocked: int = 0  # пропущено через robots.txt
    nbytes: int = 0
    started: float = field(default_factory=time.perf_counter)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def pages_per_second(self) -> float:
        return self.pages / self.elapsed if self.elapsed > 0 else 0.0
