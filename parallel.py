from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Tuple, TypeVar, Iterator, List

T = TypeVar('T')
R = TypeVar('R')


def p_reduce(f: Callable[[Tuple[T, T]], R], it: Iterator[T]) -> R:
    """
    it gain performance when it size >= 1000
    """
    pool = ThreadPoolExecutor()
    while len(it) != 1:
        if len(it) & 1:
            it, remain = it[:-1], it[-1:]
        else:
            remain = []
        ls_gps = list(zip(
            [it[i] for i in range(0, len(it), 2)],
            [it[i] for i in range(1, len(it), 2)]
        ))
        it = list(pool.map(f, ls_gps)) + remain
    return it[0]

