import collections
import itertools
from asyncio import coroutine
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Executor
from functools import reduce
from itertools import count
from typing import Union, Any, Iterable, TypeVar, Sequence, Callable, Optional, Type, Mapping

from FP import cor

T = TypeVar('T')
R = TypeVar('R')
target_type = Union[Sequence, str]


class chain:
    """
    Function invocation chain for sequence, middle operation is lazy

    Examples:
        .. code-block:: python

            chain(obj).map(...).filter(...).reduce(...)
    """
    __slots__ = ('__ori', '__obj', '__pool')

    def __init__(self, obj: Union[target_type, Any], pool_executor: Executor = None):
        """
        :param obj: support Union[list, tuple, str], if not will try to parse obj to list
        :param pool_executor: let map run in pool executor, if None then run linearly
        :raise TypeError: cannot parse obj into list
        """
        self.__ori = obj
        self.__obj = obj.copy() if hasattr(obj, 'copy') else obj
        self.__pool = pool_executor

    def _set_obj(self, obj: Union[target_type, Any]):
        """
        protected method only for ExecutorFhChain to allow reuse of Executor

        :param obj: support Union[list, tuple, str], if not will try to parse obj to list
        :raise TypeError: cannot parse obj into list
        """
        self.__ori = obj
        self.__obj = obj.copy() if hasattr(obj, 'copy') else obj

    # collect method
    def foldl(self, f: Callable[[target_type], T], init: T = None) -> T:
        """
        fold sequence from left, show as fig below

        f ----- Sequence -----> result

        (x + y) ----- [1,2,3,4,5] ----- (1+2) [3,4,5] ----- (3+3) [4,5] ----- ... -----> 15

        (x + y) ----- init=5 ----- (5+1) [2,3,4,5] -----...-----> 20

        :param f: (Sequence -> T)
        :param init: seed of the whole folding process, if None then start from Sequence[0]
        :return: result after applied function to whole sequence
        """
        if init is not None:
            self.__obj = reduce(f, self.__obj, init)
        else:
            self.__obj = reduce(f, self.__obj)
        return self.__obj

    def reduce(self, f: Callable[[target_type], T], init: T = None) -> T:
        """
        alias of ``foldl``
        """
        return self.foldl(f, init)

    def foldr(self, f: Callable[[target_type], T], init: T = None) -> T:
        """
        like ``foldl`` but fold from the end of Sequence, which is fold right
        """
        # reversed foldl
        self.__obj = reversed(self.__obj)
        return self.foldl(f, init)

    def all(self, f: Callable[[T], bool] = None) -> bool:
        """
        Check if all items in Sequence match ``f(x)`` condition or not

        either one item False -> False

        if ``f`` is None, all will be applied to Sequence directly

        :param f: Optional[(T -> bool)]
        """
        if not f:
            return all(self.__obj)
        else:
            return all(self.map(lambda x: f(x)).it())

    def any(self, f: Callable[[T], bool] = None) -> bool:
        """
        Check if any items match ``f(x)``

        either one item True -> True

        if ``f`` is None, any will be applied to Sequence directly

        :param f: Optional[(T -> bool)]
        """
        if not f:
            return any(self.__obj)
        else:
            return any(self.map(lambda x: f(x)).it())

    def first(self, f: Callable[[T], bool]) -> Optional[T]:
        """
        :param f: (T -> bool)
        :return: first item which match f(x), else None
        """
        try:
            return next(filter(lambda x: f(x), self.__obj))
        except StopIteration:
            return None

    def first_index(self, f: Callable[[T], bool]) -> Optional[int]:
        """
        :param f: (T -> bool)
        :return: first item index which match f(x), else None
        """
        try:
            return next(map(lambda x: (f(x[0]), x[1]), zip(self.__obj, count(start=0, step=1))))[1]
        except StopIteration:
            return None
        # i = 0
        # for item in self.__obj:
        #     if f(item):
        #         return i
        #     i += 1
        # return None

    def first_not(self, f: Callable[[T], bool]) -> Optional[T]:
        """
        :param f: (T -> bool)
        :return: first item which not match f(x), else None
        """
        try:
            return next(filter(lambda x: not f(x), self.__obj))
        except StopIteration:
            return None
        # self.filter(lambda x: not f(x))
        # for ret in self.__obj:
        #     return ret
        # return None

    def first_not_index(self, f: Callable[[Any], bool]) -> Optional[int]:
        """
        :param f: (T -> bool)
        :return: first item index which not match f(x), else None
        """
        try:
            return next(map(lambda x: (not f(x[0]), x[1]), zip(self.__obj, count(start=0, step=1))))[1]
        except StopIteration:
            return None
        # i = 0
        # for item in self.__obj:
        #     if not f(item):
        #         return i
        #     i += 1
        # return None

    def indexes(self, f: Callable[[T], bool]) -> Optional[Sequence[int]]:
        """
        :param f: (T -> bool)
        :return: items' index which match f(x)
        """
        try:
            return reduce(lambda x, y: x + [y], filter(lambda x: f(x[0]), zip(self.__obj, count(start=0, step=1))), [])
        except StopIteration:
            return None
        # i = 0
        # ret = []
        # for item in self.__obj:
        #     if f(item):
        #         ret.append(i)
        #     i += 1
        # return ret

    def distinct(self) -> bool:
        """
        :return: whole sequence contains same element?
        """
        return len(set(self.__obj)) != 1

    def count(self) -> int:
        """
        :return: sequence length
        """
        return reduce(lambda x, _: x + 1, self.__obj, 0)

    def max(self) -> T:
        """
        :return: max item in sequence
        """
        return max(self.__obj)

    def min(self) -> T:
        """
        :return: min item in sequence
        """
        return min(self.__obj)

    def sum(self) -> T:
        """
        alias of ``reduce(lambda x, y: x+y)``

        :return: summation result
        """
        return sum(self.__obj)

    def average(self) -> Union[int, float]:
        """
        ``sum / count``, required Sequence type is divisible

        In practise, ``int`` or ``float`` will be returned

        :return: avg in int or float
        """
        return self.sum() / self.count()

    def execute(self) -> None:
        """
        do nothing, just iterate obj once

        :return: None
        :raise PicklingError: Only raise when pool is ProcessPoolExecutor, f must global function to avoid this
        """
        from .common import identity
        if self.__pool:
            [_ for _ in self.__pool.map(identity, self.__obj)]
        else:
            [_ for _ in self.__obj]
        return None

    def collect(self, t: Type = list) -> target_type:
        """
        collect all items of sequence and cast result as Type ``t``

        :param t: type which used to cast result, in practice list will be used
        :return: result as Type ``t``
        """
        return t(self.__obj)

    def split(self, f: Callable[[T], bool]) -> Sequence[target_type]:
        """
        split sequence into sublist by ``f(x)``, like ``str.split()``, splitter itself will be vanished

        :param f: (T -> bool)
        :return: sublist of T
        """
        ret = []
        tmp = []
        for o in self.__obj:
            if f(o):
                ret.append(tmp)
                tmp = []
            else:
                tmp.append(o)
        ret.append(tmp)
        return ret

    def split_by_index(self, *index: int) -> Sequence[target_type]:
        """
        split sequence into sublist by ``index``, index item will not be vanished

        :param index: location you want to split sequence, eg: 1 -> [0], [1:]
        :return: sublist of T
        """
        index = sorted(set(index))
        ret = []
        tmp = []
        count = 0
        for o in self.__obj:
            if count in index:
                ret.append(tmp)
                tmp = [o]
            else:
                tmp.append(o)
            count += 1
        ret.append(tmp)
        return ret

    def flatten(self, layer: int = 1) -> Sequence:
        """
        flatten Sequence

        Examples:
            .. code-block:: python

                flatten([[1,2],[3,4]]) = [1,2,3,4]

        :raise TypeError: If Sequence layer is irregular like [1, [2, [3], 4, [5, 6, [7]]]],
            layer >= 1 may raise TypeError
        :param layer: <=0 -> flatten all layer, >= 1 flatten specific layer
        :return: Sequence
        """
        ret = self.__obj
        if layer >= 1:
            for _ in range(layer):
                ret = itertools.chain.from_iterable(ret)
        else:
            def core(ls):
                cache = []
                for l in ls:
                    if isinstance(l, (list, set, tuple)):
                        l2 = core(l)
                        for l2_item in l2:
                            cache.append(l2_item)
                    else:
                        cache.append(l)
                return cache

            tmp = []
            for o in ret:
                if isinstance(o, (list, tuple, set)):
                    ls = list(core(o))
                    for l in ls:
                        tmp.append(l)
                else:
                    tmp.append(o)
            ret = tmp
        return list(ret)

    def group_by(self,
                 key: Optional[str] = None,
                 attr: Optional[str] = None,
                 f: Optional[Callable[[Any], Any]] = None) -> Mapping:
        """
        group Sequence by Union[key, attr, f]

        Priority: Key > attr > f

        :raise ValueError: no params is provided, key is not exist in item
        :raise TypeError: item is not Union[Sequence, Mapping] and use key to filter
        :raise AttributeError: attr is not exist in item
        :param key: key of item (indicate item: Union[Sequence, Mapping])
        :param attr: attr of item
        :param f: function that map to item as group filter
        :return: Mapping of grouping condition (key,attr or f) to realted value
        """
        if key is None and f is None and attr is None:
            raise ValueError('Either f, attr or key should be set')
        ret = {}  # value => obj
        if key:
            for o in self.__obj:
                target = o[key]
                if target not in ret:
                    ret[target] = []
                ret[target].append(o)
        elif attr:
            for o in self.__obj:
                target = getattr(o, attr)
                if target not in ret:
                    ret[target] = []
                ret[target].append(o)
        else:
            for o in self.__obj:
                target = f(o)
                if target not in ret:
                    ret[target] = []
                ret[target].append(o)
        return ret

    # pipe method
    def map(self, f: Callable[[T], R]) -> 'chain':
        """
        apply ``f`` on item and create a new copy on mapped sequence

        original sequence will not be affected

        lazy evaluated

        :param f: (T -> R)
        :raise PicklingError: Only raise when pool is ProcessPoolExecutor, f must global function to avoid this
        """
        if not self.__pool:
            self.__obj = map(f, self.__obj)
        else:
            self.__obj = self.__pool.map(f, self.__obj)
        return self

    def filter(self, f: Callable[[T], bool]) -> 'chain':
        """
        filter out items which match ``f(x)``

        original sequence will not be affected

        lazy evaluated

        :param f: (T -> bool)
        """
        self.__obj = filter(f, self.__obj)
        return self

    def non_empty(self) -> 'chain':
        """
        filter out non_empty value, may mis-filter inappropriate items

        alias of ``filter(lambda x: x)``

        original sequence will not be affected

        lazy evaluated
        """
        return self.filter(lambda x: x)

    def empty(self) -> 'chain':
        """
        filter out empty items, may mis-filter inappropriate items

        alias of ``filter(lambda x: not x)``

        original sequence will not be affected

        lazy evaluated
        """
        return self.filter(lambda x: not x)

    def apply(self, f: Callable[[T], R]) -> 'chain':
        """
        apply function to item but do not affect item value, just execute without mapping

        original sequence will not be affected

        lazy evaluated

        :param f: (T -> R)
        """

        def body(item):
            i_cp = item if not hasattr(item, 'copy') else item.copy()
            f(i_cp)
            return item

        return self.map(body)

    def sorted(self, key: str = None, attr: str = None, f: Callable = None) -> 'chain':
        """
        sort Sequence by Union[key, attr, f]

        Priority: key > attr > f

        :raise ValueError: no params is provided, key is not exist in item
        :raise TypeError: item is not Union[Sequence, Mapping] and use key to filter
        :raise AttributeError: attr is not exist in item
        :param key: key of item (indicate item: Union[Sequence, Mapping])
        :param attr: attr of item
        :param f: function that map to item as sort filter
        :return: chain with sorted sequence
        """
        if key is None and f is None and attr is None:
            raise ValueError('Either f, attr or key should be set')
        self.__obj = sorted(self.__obj,
                            key=lambda x: x[key] if key is not None else getattr(x, attr) if attr is not None else f)
        return self

    # special function
    def it(self, wrap: bool = False) -> Union['chain', Iterable]:
        """
        ``wrap``:

        True -> new instance of chain with sequence operated at current stage

        False -> non-executed iterable
        """
        ret = self.__obj.copy() if hasattr(self.__obj, 'copy') else self.__obj
        if not wrap:
            return ret
        return chain(ret)

    def ori(self) -> target_type:
        """
        :return: original sequence at the beginning
        """
        return self.__ori

    def restore(self) -> 'chain':
        """
        reset the Sequence to original one
        """
        self.__obj = self.__ori.copy() if hasattr(self.__ori, 'copy') else self.__ori
        return self

    def to_list(self) -> list:
        return list(self.__obj)

    def to_tuple(self) -> tuple:
        return tuple(self.__obj)

    def to_set(self) -> set:
        return set(self.__obj)

    def to_dict(self) -> dict:
        return dict(self.__obj)

    def print(self, msg) -> 'chain':
        return self.apply(lambda _: print(msg))

    def __iter__(self):
        return self

    def __next__(self):
        if not hasattr(self.__obj, '__iter__') or not hasattr(self.__obj, '__next__'):
            self.__obj = iter(self.__obj)
        return next(self.__obj)

    def __aiter__(self):
        return self

    def __anext__(self):
        if not hasattr(self.__obj, '__iter__') or not hasattr(self.__obj, '__next__'):
            self.__obj = iter(self.__obj)
        try:
            return next(self.__obj)
        except StopIteration:
            raise StopAsyncIteration


class thread_chain(chain):
    __slots__ = ('__ori', '__obj', '__pool')

    def __init__(self, obj: Union[target_type, Any] = None):
        super().__init__(obj, ThreadPoolExecutor())

    def set_obj(self, obj: Union[target_type, Any]):
        self._set_obj(obj)
        return self


class process_chain(chain):
    """
    CAUTION!!!

    All function pass to this chain must be global function, which is "using def on top modules"
    Else PicklingError will be raised
    """
    __slots__ = ('__ori', '__obj', '__pool')

    def __init__(self, obj: Union[target_type, Any] = None):
        super().__init__(obj, ProcessPoolExecutor())

    def set_obj(self, obj: Union[target_type, Any]):
        self._set_obj(obj)
        return self


class async_chain:
    """
    This is async ver of chain, but not inherit

    this is not lazy at all
    """
    __slots__ = ('__ori', '__obj')

    def __init__(self, obj):
        self.__ori = obj
        self.__obj = obj.copy() if hasattr(obj, 'copy') else obj

    async def foldl(self, f, init=None):
        if init is not None:
            return await coroutine(reduce)(f, self.__obj, init)
        else:
            return await coroutine(reduce)(f, self.__obj)

    async def foldr(self, f, init=None):
        self.__obj = reversed(self.__obj)
        return await self.foldl(f, init)

    async def all(self, f=None):
        if not f:
            return await coroutine(all)(self.__obj)
        return await coroutine(all)(map(f, self.__obj))

    async def any(self, f=None):
        if not f:
            return await coroutine(any)(self.__obj)
        return await coroutine(any)(map(f, self.__obj))

    async def first(self, f):
        try:
            return await coroutine(next)(filter(f, self.__obj))
        except StopIteration:
            return None

    async def first_index(self, f):
        try:
            return await coroutine(next)(
                map(lambda x: (f(x[0], x[1])), zip(self.__obj, count(start=0, step=1)))
            )[1]
        except StopIteration:
            return None

    async def first_not(self, f):
        try:
            return await coroutine(next)(filter(lambda x: not f(x), self.__obj))
        except StopIteration:
            return None

    async def first_not_index(self, f):
        try:
            return await coroutine(next)(
                map(lambda x: (not f(x[0]), x[1]), zip(self.__obj, await coroutine(count)(start=0, step=1)))
            )[1]
        except StopIteration:
            return None

    async def indexes(self, f):
        try:
            return await coroutine(reduce)(
                lambda x, y: x + [y],
                filter(lambda x: f(x[0]), zip(self.__obj, count(start=0, step=1)))
                , [])
        except StopIteration:
            return None

    async def distinct(self):
        return await coroutine(len)(set(self.__obj)) != 1

    async def count(self):
        return await coroutine(reduce)(lambda x, _: x + 1, self.__obj, 0)

    async def max(self):
        return await coroutine(max)(self.__obj)

    async def min(self):
        return await coroutine(min)(self.__obj)

    async def sum(self):
        return await coroutine(sum)(self.__obj)

    async def average(self):
        return await self.sum() / await self.count()

    async def execute(self):
        await coroutine(list)(self.__obj)
        return None

    async def collect(self, t):
        return await coroutine(t)(self.__obj)

    async def split(self, f):
        ret = []
        tmp = []
        for o in self.__obj:
            if await coroutine(f)(o):
                await coroutine(ret.append)(tmp)
                tmp = []
            else:
                await coroutine(tmp.append)(o)
        if tmp:
            await coroutine(ret.append)(tmp)
        return ret

    async def split_by_index(self, *idx):
        idx = await coroutine(sorted)(set(idx))
        ret = []
        tmp = []
        count = 0
        for o in self.__obj:
            if count in idx:
                await coroutine(ret.append)(tmp)
                tmp = [o]
            else:
                await coroutine(tmp.append)(o)
            count += 1
        if tmp:
            await coroutine(ret.append)(tmp)
        return ret

    async def flatten(self, layer=1):
        ret = self.__obj
        if layer >= 1:
            for _ in await coroutine(range)(layer):
                ret = await coroutine(itertools.chain.from_iterable)(ret)
        else:
            async def core(ls):
                cache = []
                for l in ls:
                    if await coroutine(isinstance)(l, (list, set, tuple)):
                        l2 = await core(l)
                        for l2_item in l2:
                            await coroutine(cache.append)(l2_item)
                    else:
                        await coroutine(cache.append)(l)
                return cache

            tmp = []
            for o in ret:
                if await coroutine(isinstance)(o, (list, set, tuple)):
                    ls = await coroutine(list)(await core(o))
                    for l in ls:
                        await coroutine(tmp.append)(l)
                else:
                    await coroutine(tmp.append)(o)
            ret = tmp
        return await coroutine(list)(ret)

    async def group_by(self, key, attr, f):
        if key is None and f is None and attr is None:
            raise ValueError('Either f, attr or key should be set')
        ret = {}
        if key:
            for o in self.__obj:
                target = await coroutine(o.__getitem__)(key)
                if target not in ret:
                    await coroutine(ret.__setitem__)(target, [])
                await coroutine(await coroutine(ret.__getitem__)(target).append)(o)
        elif attr:
            for o in self.__obj:
                target = await coroutine(getattr)(o, attr)
                if target not in ret:
                    await coroutine(ret.__setitem__)(target, [])
                await coroutine(await coroutine(ret.__getitem__)(target).append)(o)
        else:
            for o in self.__obj:
                target = await coroutine(f)(o)
                if target not in ret:
                    await coroutine(ret.__setitem__)(target, [])
                await coroutine(await coroutine(ret.__getitem__)(target).append)(o)
        return ret

    async def map(self, f):
        self.__obj = await coroutine(list)(map(f, self.__obj))
        return self

    async def filter(self, f):
        self.__obj = await coroutine(list)(filter(f, self.__obj))
        return self

    async def non_empty(self):
        return await self.filter(lambda x: x)

    async def empty(self):
        return await self.filter(lambda x: not x)

    async def apply(self, f):
        def body(item):
            i_cp = item if not hasattr(item, 'copy') else item.copy()
            f(i_cp)
            return item

        return await self.map(body)

    async def sorted(self, key, attr, f):
        if key is None and f is None and attr is None:
            raise ValueError('Either f, attr or key should be set')
        self.__obj = await coroutine(sorted)(
            self.__obj,
            key=lambda x: x[key] if key is not None else getattr(x, attr) if attr is not None else f
        )
        return self

    async def it(self, wrap=False):
        ret = await coroutine(self.__obj.copy)() if hasattr(self.__obj, 'copy') else self.__obj
        if not wrap:
            return ret
        return async_chain(ret)

    def ori(self):
        return self.__ori

    async def restore(self):
        self.__obj = await coroutine(self.__ori.copy)() if hasattr(self.__ori, 'copy') else self.__ori
        return self

    async def to_list(self) -> list:
        return await coroutine(list)(self.__obj)

    async def to_tuple(self) -> tuple:
        return await coroutine(tuple)(self.__obj)

    async def to_set(self) -> set:
        return await coroutine(set)(self.__obj)

    async def to_dict(self) -> dict:
        return await coroutine(dict)(self.__obj)

    async def print(self, msg):
        return await self.apply(lambda _: print(msg))
