from enum import auto, Flag
from functools import reduce
from typing import Iterable, Sequence, TypeVar, Callable, Any, Union, Mapping, Tuple, List, Coroutine, Hashable, Type, \
    Optional, Dict

from .compose import compose
from .curry import curry
from .trampoline import trampoline

T = TypeVar('T')
U = TypeVar('U')
R = TypeVar('R')
Number = TypeVar('Number', int, float)


@curry
def pos(idx: Hashable, collection: Union[Sequence[T], Mapping[Hashable, T]]) -> T:
    """
    return item on specific position of collections

    :param idx: index, any hashable type is valid key
    :param collection: collection to be searched
    :return: item/None
    """
    try:
        return collection[idx]
    except (KeyError, IndexError):
        return None


def fst() -> Callable[[Union[Sequence[T], Mapping[Hashable, T]]], T]:
    """
    alias of pos(0)
    """
    return pos(0)


def snd() -> Callable[[Union[Sequence[T], Mapping[Hashable, T]]], T]:
    """
    alias of pos(1)
    """
    return pos(1)


@curry
def c_filter_lazy(f: Callable[[T], bool], it: Iterable[T]) -> Iterable[T]:
    """
    curried filter
    :param f: function used to filter
    :param it: iterable
    :return: filtered generator
    """
    return filter(f, it)


def c_filter(t: Type = list) -> Callable[[Callable[[T], bool], Iterable[T]], Iterable[T]]:
    """
    strict version of c_filter_lazy
    """
    return compose(t, c_filter_lazy)


@curry
def c_map_lazy(f: Callable[[T], U], it: Iterable[T]) -> Iterable[U]:
    """
    curried map
    :param f: function used to map
    :param it: iterable
    :return: mapped iterable generator
    """
    return map(f, it)


def c_map(t: Type = list) -> Callable[[Callable[[T], U], Iterable[T]], Iterable[U]]:
    """
    strict version of c_map_lazy
    """
    return compose(t, c_map_lazy)


@curry
def c_mapif_lazy(f_map: Callable[[T], U], f_filter: Callable[[T], bool], it: Iterable[T]) -> Iterable[U]:
    """
    curried filter->map->result
    :param f_map: function use in map
    :param f_filter: function use in filter
    :param it: iterable
    :return: handled iterable
    """
    return map(f_map, filter(f_filter, it))


def c_mapif(t: Type = list) -> Callable[[Callable[[T], U], Callable[[T], bool], Iterable[T]], Iterable[U]]:
    """
    strict version of c_mapif_lazy
    """
    return compose(t, c_mapif_lazy)


def head(xs: Sequence[T]) -> T:
    """
    alias of fst
    """
    try:
        return xs[0]
    except IndexError:
        return None


def tail(xs: Sequence[T]) -> Sequence[T]:
    """
    return slice of input sequence except first item
    :param xs: sequence
    :return: sequence without first item
    """
    return xs[1:]


@curry
def take(n: int, xs: Sequence[T]) -> Sequence[T]:
    """
    :return: first n item as list
    """
    return xs[0:n]


@curry
def drop(n: int, xs: Sequence[T]) -> Sequence[T]:
    """
    :return: slice which start from *n* to end of *xs*
    """
    return xs[n:]


def last(xs: Sequence[T]) -> T:
    """
    :return: last item of *xs* or None
    """
    try:
        return xs[-1]
    except IndexError:
        return None


def init(xs: Sequence[T]) -> Sequence[T]:
    """
    :return: list without last item of *xs*
    """
    return xs[:-1]


def identity(x: T) -> T:
    """
    :return *x*, do nothing
    """
    return x


@curry
def const(a: T, _: Any) -> T:
    """
    :return: only return *a*
    """
    return a


@curry
def foldl(f: Callable[[T], R], init: R, xs: Sequence[T]) -> R:
    """
    :return: fold from left, which is alias of reduce with init param
    """
    return reduce(f, xs, init)


@curry
def foldl1(f: Callable[[T], R], xs: Sequence[T]) -> R:
    """
    :return: fold from left with default init as *fst(xs)*
    """
    return reduce(f, xs)


@curry
def foldl_if(f_fold: Callable[[T], R], f_filter: Callable[[T], bool], init: R, xs: Sequence[T]) -> R:
    """
    :return: filter->foldl
    """
    return reduce(f_fold, filter(f_filter, xs), init)


@curry
def foldr(f: Callable[[T], R], init: R, xs: Sequence[T]) -> R:
    """
    :return: fold from right, which is alias of *foldl(..., reversed(xs))*
    """
    return reduce(f, reversed(xs), init)


@curry
def foldr1(f: Callable[[T], R], xs: Sequence[T]) -> R:
    """
    :return: like foldl1, just start from the end, i.e. reversed
    """
    return reduce(f, reversed(xs))


@curry
def foldr_if(f_fold: Callable[[T], R], f_filter: Callable[[T], bool], init: R, xs: Sequence[T]) -> R:
    """
    :return: filter->foldr
    """
    return reduce(f_fold, filter(f_filter, reversed(xs)), init)


def concat(xss: Iterable[Iterable[T]]) -> Sequence[T]:
    """
    :return: flatten list, only flatten one layer
    """
    return [x for xs in xss for x in xs]


@curry
def exec_then_it(f: Callable[[T], Any], x: T) -> T:
    """
    :param f: callable that use x, SHOULD BE pure function
    :param x: param
    :return: x
    """
    f(x)
    return x


def print_identity(x: T) -> T:
    """
    Impure

    for DEBUG, print item and return item itself
    """
    return exec_then_it(print, x)


def collect(it: Iterable[T]) -> List[T]:
    """
    alias of *list*
    """
    return list(it)


@curry
def duplicate(n: int, o: T) -> List[T]:
    """
    :return: duplicate object *o* for *n* times as a list
    """
    return [o for _ in range(n)]


@curry
def replace_key(d: dict, old_new_pair: List[Tuple[str, str]]) -> dict:
    """
    replace old key by new key
    """
    olds = list(map(fst, old_new_pair))
    ret = {x: y for x, y in d.items() if x not in olds}
    for old, new in old_new_pair:
        ret[new] = d[old]
    return ret


@trampoline(3)
@curry
def fix(c: Callable[[T, R], bool], f: Callable[[T], R], x: T) -> R:
    """
    Theory: when reach fix point i.e. f(x) = x, stop exec f, else keep iterating

    :return: fix point value
    """
    if c(x, f(x)):
        return x
    return fix(c, f, f(x))


@curry
def elem(x: T, xs: Sequence[T]) -> bool:
    """
    :return: alias of *x in xs*
    """
    return x in xs


def flatten() -> Callable[[Iterable[Iterable[T]]], Sequence[T]]:
    """
    alias of concat
    """
    # this only flatten one layer, for multi layer flatten please use chain or fix->flatten
    # eg: fix(eq, flatten, [1, [2, [3, 4], [5, [6, 7], 8], 9], 10])
    return concat


async def cor(f: Callable) -> Coroutine:
    """
    async function wrapper
    """
    from asyncio import coroutine
    return coroutine(f)


@curry
def attr(attr_name: str, x: Any) -> Optional[Any]:
    """
    alias of getattr, default None
    """
    return getattr(x, attr_name, None)


@curry
def attr_default(attr_name: str, default: Any, x: Any) -> Any:
    """
    alias of getattr
    """
    return getattr(x, attr_name, default)


@curry
def getitem(key: Hashable, d: Dict[Hashable, Any]) -> Optional[Any]:
    """
    safe get dict value with default None
    """
    return d.get(key, None)


@curry
def getitem_default(key: Hashable, default: Any, d: Dict[Hashable, Any]) -> Any:
    """
    safe get dict value
    """
    return d.get(key, default)


class SplitTargetPolicy(Flag):
    APPEND_AT_PREVIOUS = auto()
    START_AT_NEXT = auto()
    IGNORE = auto()


@curry
def split_by_fn(f: Callable[[T], bool], target_policy: SplitTargetPolicy, xs: Iterable[T]) -> List[List[T]]:
    ret = []
    tmp = []
    for x in xs:
        if not f(x):
            tmp.append(x)
        else:
            if target_policy & SplitTargetPolicy.APPEND_AT_PREVIOUS:
                tmp.append(x)
            ret.append(tmp)
            tmp = [x] if target_policy & SplitTargetPolicy.START_AT_NEXT else []
    if tmp:
        ret.append(tmp)
    return ret


@curry
def split_by_attr(
        attribute: str, compare_f: Callable[[T], bool], target_policy: SplitTargetPolicy, xs: Iterable[T]
) -> List[List[T]]:
    return split_by_fn(lambda x: compare_f(getattr(x, attribute)), target_policy, xs)


@curry
def split_by_key(
        key: str, compare_f: Callable[[T], bool], target_policy: SplitTargetPolicy, xs: Iterable[T]
) -> List[List[T]]:
    return split_by_fn(lambda x: compare_f(x[key]), target_policy, xs)
