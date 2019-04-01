from typing import Union, Type, Sequence, Callable, TypeVar

from .common import curry
from .trampoline import trampoline

T = TypeVar('T')


@curry
def mod(a, b):
    return a % b


@curry
def floordiv(a, b):
    return a // b


@curry
def add(a, b):
    return a + b


@curry
def lt(a, b):
    return a < b


@curry
def le(a, b):
    return a <= b


@curry
def eq(a, b):
    return a == b


@curry
def ne(a, b):
    return a != b


@curry
def ge(a, b):
    return a >= b


@curry
def gt(a, b):
    return a > b


@curry
def shl(a, b):
    return a << b


@curry
def mul(a, b):
    return a * b


def neg(a):
    return -a


@curry
def shr(a, b):
    return a >> b


@curry
def pow(a, b):
    return a ** b


@curry
def sub(a, b):
    return a - b


@curry
def truediv(a, b):
    return a / b


def not_(a):
    return not a


# inverse version
@curry
def inverse_exec(f, b, a):
    return f(a, b)


imod = inverse_exec(mod)
ifloordiv = inverse_exec(floordiv)
ilt = inverse_exec(lt)
ile = inverse_exec(le)
ige = inverse_exec(ge)
igt = inverse_exec(gt)
ishl = inverse_exec(shl)
ishr = inverse_exec(shr)
ipow = inverse_exec(pow)
isub = inverse_exec(sub)
itruediv = inverse_exec(truediv)


# bit manipulation
@curry
def b_and(a, b):
    return a & b


@curry
def b_or(a, b):
    return a | b


@curry
def xor(a, b):
    return a ^ b


def b_not(a):
    return ~a


# Matrix use
@curry
def matmul(a, b):
    return a @ b


@curry
def is_(t, a):
    return a is t


@curry
def ins(t: Union[Type, Sequence[Type]], o):
    return isinstance(o, t)


@trampoline()
def not_f(f: Callable[[T], bool]):
    def core(*args, **kwargs):
        result = f(*args, **kwargs)
        if not type(result) == bool:  # maybe curry or partial function
            return not_f(result)
        return not result

    return core
