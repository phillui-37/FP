import functools
import sys


class TrampolineException(Exception):
    __slots__ = ('args', 'kwargs')

    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


def trampoline(layers=2):
    """
    normally this fn is used as decorator on fn tco, which layers should be 2

    1.

    Example:

    @trampoline()
    def foo(i: int):
        return i-1

    Execution stack:

    wrapper -> foo (count: 2)

    2.
    But this the stack thickness vary with fn wrapped by trampoline

    Example:
    @trampoline(3)
    @curry
    def bar(x: int, y: int):
        return x+y

    Execution stack:

    wrapper -> curry -> bar


    i.e. You should know the stuffing layers thickness, change layers value yourself
    """

    def core(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            frame = sys._getframe()

            def get_target_and_verify(fm):
                target = fm.f_back
                for _ in range(layers, 0, -1):
                    if not target:
                        return False
                    if target.f_code == fm.f_code:
                        return True
                    target = target.f_back
                return False

            if get_target_and_verify(frame):
                raise TrampolineException(args, kwargs)
            else:
                while True:
                    try:
                        return f(*args, **kwargs)
                    except TrampolineException as e:
                        args = e.args
                        kwargs = e.kwargs

        return wrapper

    return core
