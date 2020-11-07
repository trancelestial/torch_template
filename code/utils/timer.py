from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Optional
import time
from datetime import timedelta

class TimerError(Exception):
    '''A custom exception used to report errors in use of Timer class'''

@dataclass
class Timer(ContextDecorator):
    timers: ClassVar[Dict[str, float]] = dict()
    name: Optional[str] = None
    text: str = 'Elapsed time: {}'
    format_fnct: Optional[Callable[[int], None]] = lambda t : str(timedelta(seconds=t)).split('.')[0]
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.name:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        '''Start a new timer'''
        if self._start_time is not None:
            raise TimerError(f'Timer is running. Use .stop() to stop it')

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        '''Stop the timer, and report the elapsed time'''
        if self._start_time is None:
            raise TimerError(f'Timer is not running. Use .start() to start it')

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        if self.logger:
            self.logger(self.text.format(self.format_fnct(elapsed_time)))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time

    def __enter__(self) -> 'Timer':
        '''Start a new timer as a context manager'''
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        '''Stop the context manager timer'''
        self.stop()