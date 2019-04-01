from collections import OrderedDict
from copy import deepcopy
from inspect import signature, Parameter
from typing import Callable, Union


class curry:
    """
    Automated partial function which work once only

    Works on both lambda and normal generator/function

    To support annotation, args and kwargs is not allowed in __init__

    To support class method, self/cls must be passed in manually as curry won't handle this for you

    Not work on built-in fuction which do not contains function signature

    Not support *args and **kwargs

    Examples:

        .. code-block:: python

            a = curry(lambda x,y,z: x+y+z, False)
            b = a(2)
            c = b(2,3) # 7
            d = b(5,6) # Exception thrown
            # since curry only serve one invocation only, extra curry required if derive the function again
            e = curry(a(2))
            f = e(3,5) # 10
            f = e(10,10) # works now, 22

        .. code-block:: python

            # for generator, auto_eval is used
            @curry
            def a(x,y,z)
                yield x+y+z
            b = a(2)
            c = b(2,3) # 7
            # the following like the previous example...

    """

    __slots__ = ('f', 'default_args_queue', 'sig', 'args_queue')

    def __init__(self, f: Union[Callable, 'curry']):
        # assume f does not contain varargs
        if isinstance(f, curry):
            self.f = f.f
            self.default_args_queue = f.default_args_queue
            self.sig = f.sig
            self.args_queue = f.args_queue
        else:
            self.f = f
            self.sig = signature(f)
            self.default_args_queue = OrderedDict()
            for k, v in self.sig.parameters.items():
                if v.default != Parameter.empty:
                    self.default_args_queue[k] = v.default
            self.args_queue = {}

    def __call__(self, *args, **kwargs):
        new_instance = deepcopy(self)
        f_sig_keys = new_instance.sig.parameters.keys()
        if args:
            for arg in args:
                # try:
                key = next(filter(lambda x: x not in new_instance.args_queue, f_sig_keys))
                new_instance.args_queue[key] = arg
        # except StopIteration:
        #     pass
        if kwargs:
            for k, v in kwargs.items():
                if k not in new_instance.sig.parameters:
                    raise ValueError('{} is not a valid param key'.format(k))
                new_instance.args_queue[k] = v
        params = {**new_instance.default_args_queue, **new_instance.args_queue}
        params_len_diff = len(new_instance.sig.parameters) - len(params)
        if params_len_diff == 0:
            return new_instance.f(**params)
        elif params_len_diff < 0:
            raise Exception('Params len exceed!')
        else:
            return curry(new_instance)

