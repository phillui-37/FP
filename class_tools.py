class SealedClassMeta(type):
    def __new__(mcs, name, bases, attrs):
        for b in bases:
            if isinstance(b, SealedClassMeta):
                raise TypeError('{} cannot be inherited.'.format(b.__name__))
            return type.__new__(mcs, name, bases, dict(attrs))


def is_case(o: object, t: type) -> bool:
    return o.__metaclass__ is t


class enum_wrapper:
    __slots__ = ('__entries',)

    def __init__(self, ins):
        self.__entries = list(ins)

    def __getattr__(self, item: str):
        if not item.startswith('_'):
            return next(filter(lambda x: x.name == item, self.__entries)).value
        return getattr(self, item)
