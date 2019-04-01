from functools import reduce
from typing import Callable, Union


class compose:
    """
    Compose functions into a pipeline, which the result will pass over functions

    Function invocation order: left->right(default)

    Examples:

        .. code-block:: python

            # order is filter->map
            compose(partial(map, lambda x,y,z: x+y+z)) | partial(filter, lambda x: x & 1 == 0)
            # order is map->filter
            partial(filter, lambda x: x & 1 == 0) | compose(partial(map, lambda x,y,z: x+y+z))
            # order is map->filter
            compose(partial(map, lambda x,y,z: x+y+z), from_left=False) | partial(filter, lambda x: x & 1 == 0)
    """
    __slots__ = ('__from_left', '__f_list')

    def __init__(self, *fs: Callable, from_left: bool=False):
        self.__from_left = from_left
        self.__f_list = list(fs)

    def __call__(self, *args, **kwargs):
        """
        [f1...fn]
        from_left -> FIFO(pipe) -> fn(...f2(f1(x)))
        from_right -> FILO(compose) -> f1(...fn(x))

        so for compose, fn list will be reversed before invoking
        """
        fs = self.__f_list if self.__from_left else list(reversed(self.__f_list))
        return reduce(lambda result, f: f(result), fs[1:], fs[0](*args, **kwargs))

    def __or__(self, other: Union[Callable, 'compose']):
        # other must be callable, you know that
        if isinstance(other, compose):
            self.__f_list = [*self.__f_list, *other.get_fn_list]
        else:
            self.__f_list = [*self.__f_list, other]
        return self

    def __ror__(self, other: Union[Callable, 'compose']):
        if isinstance(other, compose):
            self.__f_list = [*other.get_fn_list, *self.__f_list]
        else:
            self.__f_list = [other, *self.__f_list]
        return self

    @property
    def get_fn_list(self):
        return self.__f_list.copy()
