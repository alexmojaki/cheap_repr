from __future__ import print_function, division, absolute_import

from future import standard_library

standard_library.install_aliases()

import collections
from sys import version_info

from future.utils import iteritems

from cheap_repr.utils import type_name, exception_string, safe_qualname

import inspect
import warnings
from array import array
from collections import defaultdict, deque
from importlib import import_module
from itertools import islice

PY2 = version_info[0] == 2
PY3 = not PY2


class ReprSuppressedWarning(Warning):
    """
    This warning is raised when a class is supressed from having a
    repr calculated for it by cheap_repr in the future.
    Instead the output will be of the form:
    <MyClass instance at 0x123 (repr suppressed)>
    This can happen when either:
    1. An exception is raised by either repr(x) or f(x) where
       f is a registered repr function for that class. See
       'Exceptions in repr functions' in the README for more.
    2. The output of repr(x) is longer than
       cheap_repr.suppression_threshold characters.
    """


repr_registry = {}


def try_register_repr(module_name, class_name):
    """
    This tries to register a repr function for a class that may not exist,
    e.g. if the class is in a third party package that may not be installed.
    module_name and class_name are strings. If the class can be imported,
    then:

        @try_register_repr(module_name, class_name)
        def repr_function(...)
            ...

    is equivalent to:

        from <module_name> import <class_name>
        @register_repr(<class_name>)
        def repr_function(...)
            ...

    If the class cannot be imported, nothing happens.
    """
    try:
        cls = getattr(import_module(module_name), class_name)
    except (ImportError, AttributeError):
        return lambda x: x
    else:
        if inspect.isclass(cls):
            return register_repr(cls)


def register_repr(cls):
    """
    Register a repr function for cls. The function must accept two arguments:
    the object to be represented as a string, and an instance of ReprHelper.
    The registered function will be used by cheap_repr when appropriate,
    and can be retrieved by find_repr_function(cls).
    """

    def decorator(func):
        repr_registry[cls] = func
        func.__dict__.setdefault('maxparts', 6)
        return func

    return decorator


def maxparts(num):
    """
    See the maxparts section in the README.
    """

    def decorator(func):
        func.maxparts = num
        return func

    return decorator


def basic_repr(x, *_):
    return '<%s instance at %#x>' % (type_name(x), id(x))


suppressed_classes = set()


@register_repr(object)
@maxparts(60)
def repr_object(x, helper):
    s = repr(x)
    if len(s) > cheap_repr.suppression_threshold:
        cls = x.__class__
        suppressed_classes.add(cls)
        warnings.warn(ReprSuppressedWarning(
            '%s.__repr__ is too long and has been suppressed. '
            'Register a repr for the class to avoid this warning '
            'and see an informative repr again, '
            'or increase cheap_repr.suppression_threshold' % safe_qualname(cls)))
    return helper.truncate(s)


def find_repr_function(cls):
    for cls in inspect.getmro(cls):
        func = repr_registry.get(cls)
        if func:
            return func


__raise_exceptions_from_default_repr = False


def raise_exceptions_from_default_repr():
    global __raise_exceptions_from_default_repr
    __raise_exceptions_from_default_repr = True
    repr_object.raise_exceptions = True


def cheap_repr(x, level=None):
    """
    Return a short, computationally inexpensive string
    representation of x, with approximately up to `level`
    levels of nesting.
    """
    if level is None:
        level = cheap_repr.max_level
    x_cls = getattr(x, '__class__', type(x))
    for cls in inspect.getmro(x_cls):
        if cls in suppressed_classes:
            return _basic_but('repr suppressed', x)
        func = repr_registry.get(cls)
        if func:
            helper = ReprHelper(level, func)
            return _try_repr(func, x, helper)

    # Old-style classes in Python 2.
    return _try_repr(repr, x)


cheap_repr.suppression_threshold = 300
cheap_repr.raise_exceptions = False
cheap_repr.max_level = 3


def _try_repr(func, x, *args):
    try:
        return func(x, *args)
    except BaseException as e:
        should_raise = (cheap_repr.raise_exceptions or
                        getattr(func, 'raise_exceptions', False) or
                        func is repr and __raise_exceptions_from_default_repr)
        if should_raise:
            raise
        cls = x.__class__
        if cls not in suppressed_classes:
            suppressed_classes.add(cls)
            warnings.warn(ReprSuppressedWarning(
                "Exception '%s' in %s for object of type %s. "
                "The repr has been suppressed for this type." %
                (exception_string(e), func.__name__, safe_qualname(cls))))
        return _basic_but('exception in repr', x)


def _basic_but(message, x):
    return '%s (%s)>' % (basic_repr(x)[:-1], message)


class ReprHelper(object):
    __slots__ = ('level', 'func')

    def __init__(self, level, func):
        self.level = level
        self.func = func

    def repr_iterable(self, iterable, left, right, length=None):
        if length is None:
            length = len(iterable)
        if self.level <= 0 and length:
            s = '...'
        else:
            newlevel = self.level - 1
            max_parts = self.func.maxparts
            pieces = [cheap_repr(elem, newlevel) for elem in islice(iterable, max_parts)]
            if length > max_parts:
                pieces.append('...')
            s = ', '.join(pieces)
        return left + s + right

    def truncate(self, s, middle='...'):
        max_parts = self.func.maxparts
        if len(s) > max_parts:
            i = max(0, (max_parts - 3) // 2)
            j = max(0, max_parts - 3 - i)
            s = s[:i] + middle + s[len(s) - j:]
        return s


@register_repr(tuple)
def repr_tuple(x, helper):
    if len(x) == 1:
        return '(%s,)' % cheap_repr(x[0], helper.level)
    else:
        return helper.repr_iterable(x, '(', ')')


@register_repr(list)
@try_register_repr('UserList', 'UserList')
@try_register_repr('collections', 'UserList')
def repr_list(x, helper):
    return helper.repr_iterable(x, '[', ']')


@register_repr(array)
@maxparts(5)
def repr_array(x, helper):
    if not x:
        return repr(x)
    return helper.repr_iterable(x, "array('%s', [" % x.typecode, '])')


@register_repr(set)
def repr_set(x, helper):
    if not x:
        return repr(x)
    elif PY2:
        return helper.repr_iterable(x, 'set([', '])')
    else:
        return helper.repr_iterable(x, '{', '}')


@register_repr(frozenset)
def repr_frozenset(x, helper):
    if not x:
        return repr(x)
    elif PY2:
        return helper.repr_iterable(x, 'frozenset([', '])')
    else:
        return helper.repr_iterable(x, 'frozenset({', '})')


@register_repr(collections.Set)
def repr_Set(x, helper):
    if not x:
        return '%s()' % type_name(x)
    else:
        return helper.repr_iterable(x, '%s({' % type_name(x), '})')


@register_repr(deque)
def repr_deque(x, helper):
    return helper.repr_iterable(x, 'deque([', '])')


@register_repr(dict)
@maxparts(4)
def repr_dict(x, helper):
    n = len(x)
    if n == 0:
        return '{}'
    if helper.level <= 0:
        return '{...}'
    newlevel = helper.level - 1
    pieces = []
    for key in islice(x, repr_dict.maxparts):
        keyrepr = cheap_repr(key, newlevel)
        valrepr = cheap_repr(x[key], newlevel)
        pieces.append('%s: %s' % (keyrepr, valrepr))
    if n > repr_dict.maxparts:
        pieces.append('...')
    s = ', '.join(pieces)
    return '{%s}' % (s,)


@try_register_repr('__builtin__', 'unicode')
@register_repr(str)
@maxparts(60)
def repr_str(x, helper):
    return repr(helper.truncate(x))


if PY3:
    @register_repr(bytes)
    @maxparts(60)
    def repr_bytes(x, helper):
        return repr(helper.truncate(x, middle=b'...'))


@register_repr(int)
@try_register_repr('__builtin__', 'long')
@maxparts(40)
def repr_int(x, helper):
    return helper.truncate(repr(x))


@try_register_repr('numpy.core.multiarray', 'ndarray')
def repr_ndarray(x, helper):
    if len(x) == 0:
        return repr(x)
    return helper.repr_iterable(x, 'array([', '])')


@try_register_repr('django.db.models', 'QuerySet')
def repr_QuerySet(x, _):
    try:
        model_name = x.model._meta.object_name
    except AttributeError:
        model_name = type_name(x.model)
    return '<%s instance of %s at %#x>' % (type_name(x), model_name, id(x))


@try_register_repr('collections', 'ChainMap')
@try_register_repr('chainmap', 'ChainMap')
@maxparts(4)
def repr_ChainMap(x, helper):
    return helper.repr_iterable(x.maps, type_name(x) + '(', ')')


@try_register_repr('collections', 'OrderedDict')
@try_register_repr('ordereddict', 'OrderedDict')
@try_register_repr('backport_collections', 'OrderedDict')
@maxparts(4)
def repr_OrderedDict(x, helper):
    if not x:
        return repr(x)
    helper.level += 1
    return helper.repr_iterable(iteritems(x), type_name(x) + '([', '])', length=len(x))


@try_register_repr('collections', 'UserDict')
@try_register_repr('UserDict', 'UserDict')
@register_repr(collections.Mapping)
@maxparts(5)
def repr_Mapping(x, helper):
    if not x:
        return type_name(x) + '()'
    return '{0}({1})'.format(type_name(x), repr_dict(x, helper))


@try_register_repr('collections', 'Counter')
@try_register_repr('counter', 'Counter')
@try_register_repr('backport_collections', 'Counter')
@maxparts(5)
def repr_Counter(x, helper):
    length = len(x)
    if length <= repr_Counter.maxparts:
        return repr_Mapping(x, helper)
    else:
        # The default repr of Counter gives the items in decreasing order
        # of frequency. We don't do that because it would be expensive
        # to compute. We also don't show a sample of random keys
        # because someone familiar with the default repr might be misled
        # into thinking that they are the most common.
        return '{0}({1} keys)'.format(type_name(x), length)


@register_repr(defaultdict)
@maxparts(4)
def repr_defaultdict(x, helper):
    return '{0}({1}, {2})'.format(type_name(x),
                                  x.default_factory,
                                  repr_dict(x, helper))


if PY3:
    @register_repr(type({}.keys()))
    def repr_dict_keys(x, helper):
        return helper.repr_iterable(x, 'dict_keys([', '])')


    @register_repr(type({}.values()))
    def repr_dict_values(x, helper):
        return helper.repr_iterable(x, 'dict_values([', '])')


    @register_repr(type({}.items()))
    @maxparts(4)
    def repr_dict_items(x, helper):
        helper.level += 1
        return helper.repr_iterable(x, 'dict_items([', '])')
