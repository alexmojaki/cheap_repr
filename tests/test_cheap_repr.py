import os
import re
import sys
import unittest
from array import array
from collections import defaultdict, deque
from sys import version_info, version
from unittest import skipIf

from tests.utils import TestCaseWithUtils, temp_attrs, assert_unique, Counter, skipUnless, OldStyleClass

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

try:
    from collections import ChainMap
except ImportError:
    from chainmap import ChainMap

# Python 3.9 compatibility (importing Set from collections is deprecated)
if version_info.major == 2:
    from collections import Set
else:
    from collections.abc import Set

from cheap_repr import basic_repr, register_repr, cheap_repr, PY2, PY3, ReprSuppressedWarning, find_repr_function, \
    raise_exceptions_from_default_repr, repr_registry

PYPY = 'pypy' in version.lower()


class FakeExpensiveReprClass(object):
    def __repr__(self):
        return 'bad'


register_repr(FakeExpensiveReprClass)(basic_repr)


class ErrorClass(object):
    def __init__(self, error=False):
        self.error = error

    def __repr__(self):
        if self.error:
            raise ValueError()
        return 'bob'


class ErrorClassChild(ErrorClass):
    pass


class OldStyleErrorClass:
    def __init__(self, error=False):
        self.error = error

    def __repr__(self):
        if self.error:
            raise ValueError()
        return 'bob'


class OldStyleErrorClassChild(OldStyleErrorClass):
    pass


class DirectRepr(object):
    def __init__(self, r):
        self.r = r

    def __repr__(self):
        return self.r


class RangeSet(Set):
    def __init__(self, length):
        self.length = length

    def __contains__(self, x):
        pass

    def __iter__(self):
        for x in range(self.length):
            yield x

    def __len__(self):
        return self.length


class NormalClass(object):
    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return repr(self.x)

    def foo(self):
        pass


class TestCheapRepr(TestCaseWithUtils):
    maxDiff = None

    def normalise_repr(self, string):
        string = re.sub(r'0x[0-9a-f]+', '0xXXX', string)
        string = re.sub('\\s+\n', '\n', string)
        string = re.sub('\n\n', '\n', string)
        return string

    def assert_cheap_repr(self, x, expected_repr):
        actual = self.normalise_repr(cheap_repr(x))
        self.assertEqual(actual, expected_repr)

    def assert_usual_repr(self, x, normalise=False):
        expected = repr(x)
        if normalise:
            expected = self.normalise_repr(expected)
        self.assert_cheap_repr(x, expected)

    def assert_cheap_repr_evals(self, s):
        self.assert_cheap_repr(eval(s), s)

    def assert_cheap_repr_warns(self, x, message, expected_repr):
        self.assert_warns(ReprSuppressedWarning,
                          message,
                          lambda: self.assert_cheap_repr(x, expected_repr))

    def test_registered_default_repr(self):
        x = FakeExpensiveReprClass()
        self.assertEqual(repr(x), 'bad')
        self.assert_cheap_repr(x, r'<FakeExpensiveReprClass instance at 0xXXX>')

    def test_bound_method(self):
        self.assert_usual_repr(NormalClass('hello').foo)
        self.assert_cheap_repr(
            RangeSet(10).__len__,
            '<bound method RangeSet.__len__ of RangeSet({0, 1, 2, 3, 4, 5, ...})>')

    def test_chain_map(self):
        self.assert_usual_repr(ChainMap({1: 2, 3: 4}, dict.fromkeys('abcd')))

        ex = (
            "ChainMap("
            "OrderedDict([('1', 0), ('2', 0), ('3', 0), ('4', 0), ...]), "
            "OrderedDict([('1', 0), ('2', 0), ('3', 0), ('4', 0), ...]), "
            "..., "
            "OrderedDict([('1', 0), ('2', 0), ('3', 0), ('4', 0), ...]), "
            "OrderedDict([('1', 0), ('2', 0), ('3', 0), ('4', 0), ...])"
            ")"
        ) if sys.version_info < (3, 12) else (
            "ChainMap("
            "OrderedDict({'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, ...}), "
            "OrderedDict({'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, ...}), "
            "..., "
            "OrderedDict({'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, ...}), "
            "OrderedDict({'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, ...})"
            ")"
        )
        self.assert_cheap_repr(ChainMap(*[OrderedDict.fromkeys('1234567890', 0) for _ in range(10)]),
                               ex)

    def test_list(self):
        self.assert_usual_repr([])
        self.assert_usual_repr([1, 2, 3])
        self.assert_cheap_repr([1, 2, 3] * 10 + [4, 5, 6, 7], '[1, 2, 3, ..., 5, 6, 7]')

    def test_tuple(self):
        self.assert_usual_repr(())
        self.assert_usual_repr((1,))
        self.assert_usual_repr((1, 2, 3))
        self.assert_cheap_repr((1, 2, 3) * 10 + (4, 5, 6, 7), '(1, 2, 3, ..., 5, 6, 7)')

    def test_sets(self):
        self.assert_usual_repr(set())
        self.assert_usual_repr(frozenset())
        self.assert_usual_repr(set([1, 2, 3]))
        self.assert_usual_repr(frozenset([1, 2, 3]))
        self.assert_cheap_repr(set(range(10)),
                               'set([0, 1, 2, 3, 4, 5, ...])' if PY2 else
                               '{0, 1, 2, 3, 4, 5, ...}')

    def test_dict(self):
        self.assert_usual_repr({})
        d1 = {1: 2, 2: 3, 3: 4}
        self.assert_usual_repr(d1)
        d2 = dict((x, x * 2) for x in range(10))
        self.assert_cheap_repr(d2, '{0: 0, 1: 2, 2: 4, 3: 6, ...}')
        self.assert_cheap_repr(
            {'a' * 100: 'b' * 100},
            "{'aaaaaaaaaaaaaaaaaaaaaaaaaaaa...aaaaaaaaaaaaaaaaaaaaaaaaaaaaa': 'bbbbbbbbbbbbbbbbbbbbbbbbbbbb...bbbbbbbbbbbbbbbbbbbbbbbbbbbbb'}")

        if PY3:
            self.assert_usual_repr({}.keys())
            self.assert_usual_repr({}.values())
            self.assert_usual_repr({}.items())

            self.assert_usual_repr(d1.keys())
            self.assert_usual_repr(d1.values())
            self.assert_usual_repr(d1.items())

            self.assert_cheap_repr(d2.keys(),
                                   'dict_keys([0, 1, 2, 3, 4, 5, ...])')
            self.assert_cheap_repr(d2.values(),
                                   'dict_values([0, 2, 4, 6, 8, 10, ...])')
            self.assert_cheap_repr(d2.items(),
                                   'dict_items([(0, 0), (1, 2), (2, 4), (3, 6), ...])')

    def test_defaultdict(self):
        d = defaultdict(int)
        self.assert_usual_repr(d)
        d.update({1: 2, 2: 3, 3: 4})
        self.assert_usual_repr(d)
        d.update(dict((x, x * 2) for x in range(10)))
        self.assertTrue(cheap_repr(d) in
                        ("defaultdict(%r, {0: 0, 1: 2, 2: 4, 3: 6, ...})" % int,
                         "defaultdict(%r, {1: 2, 2: 4, 3: 6, 0: 0, ...})" % int))

    def test_deque(self):
        self.assert_usual_repr(deque())
        self.assert_usual_repr(deque([1, 2, 3]))
        self.assert_cheap_repr(deque(range(10)), 'deque([0, 1, 2, 3, 4, 5, ...])')

    def test_ordered_dict(self):
        self.assert_usual_repr(OrderedDict())
        self.assert_usual_repr(OrderedDict((x, x * 2) for x in range(3)))
        self.assert_cheap_repr(OrderedDict((x, x * 2) for x in range(10)),
                               'OrderedDict([(0, 0), (1, 2), (2, 4), (3, 6), ...])'
                               if sys.version_info < (3, 12) else
                               'OrderedDict({0: 0, 1: 2, 2: 4, 3: 6, 4: 8, ...})')

    def test_counter(self):
        self.assert_usual_repr(Counter())
        self.assert_cheap_repr_evals('Counter({0: 0, 2: 1, 4: 2})')
        self.assert_cheap_repr(Counter(dict((x * 2, x) for x in range(10))),
                               'Counter(10 keys)')

    def test_array(self):
        self.assert_usual_repr(array('l', []))
        self.assert_usual_repr(array('l', [1, 2, 3, 4, 5]))
        self.assert_cheap_repr(array('l', range(10)),
                               "array('l', [0, 1, 2, ..., 8, 9])")

    def test_django_queryset(self):
        os.environ['DJANGO_SETTINGS_MODULE'] = 'tests.fake_django_settings'
        import django

        django.setup()
        from django.contrib.contenttypes.models import ContentType
        self.assert_cheap_repr(ContentType.objects.all(),
                               '<QuerySet instance of ContentType at 0xXXX>')

    if not PYPY and version_info[:2] < (3, 8):
        def test_numpy_array(self):
            import numpy

            self.assert_usual_repr(numpy.array([]))
            self.assert_usual_repr(numpy.array([1, 2, 3, 4, 5]))
            self.assert_cheap_repr(numpy.array(range(10)),
                                   'array([0, 1, 2, ..., 7, 8, 9])')

            self.assert_cheap_repr(numpy.arange(100).reshape(10, 10),
                                   """\
array([[ 0,  1,  2, ...,  7,  8,  9],
       [10, 11, 12, ..., 17, 18, 19],
       [20, 21, 22, ..., 27, 28, 29],
       ...,
       [70, 71, 72, ..., 77, 78, 79],
       [80, 81, 82, ..., 87, 88, 89],
       [90, 91, 92, ..., 97, 98, 99]])""")

            self.assert_cheap_repr(numpy.arange(1000).reshape(10, 10, 10),
                                   """\
array([[[  0,   1, ...,   8,   9],
        [ 10,  11, ...,  18,  19],
        ...,
        [ 80,  81, ...,  88,  89],
        [ 90,  91, ...,  98,  99]],
       [[100, 101, ..., 108, 109],
        [110, 111, ..., 118, 119],
        ...,
        [180, 181, ..., 188, 189],
        [190, 191, ..., 198, 199]],
       ...,
       [[800, 801, ..., 808, 809],
        [810, 811, ..., 818, 819],
        ...,
        [880, 881, ..., 888, 889],
        [890, 891, ..., 898, 899]],
       [[900, 901, ..., 908, 909],
        [910, 911, ..., 918, 919],
        ...,
        [980, 981, ..., 988, 989],
        [990, 991, ..., 998, 999]]])""")

            self.assert_cheap_repr(numpy.arange(10000).reshape(10, 10, 10, 10),
                                   """\
array([[[[   0, ...,    9],
         ...,
         [  90, ...,   99]],
        ...,
        [[ 900, ...,  909],
         ...,
         [ 990, ...,  999]]],
       ...,
       [[[9000, ..., 9009],
         ...,
         [9090, ..., 9099]],
        ...,
        [[9900, ..., 9909],
         ...,
         [9990, ..., 9999]]]])""")

            self.assert_cheap_repr(numpy.arange(128).reshape(2, 2, 2, 2, 2, 2, 2),
                                   "array(dtype('int64'), shape=(2, 2, 2, 2, 2, 2, 2))")

            self.assert_cheap_repr(numpy.ma.array([1, 2, 3], mask=[0, 1, 0]),
                                   "MaskedArray(dtype('int64'), shape=(3,))")

            self.assert_cheap_repr(numpy.matrix([[1, 2], [3, 4]]),
                                   """\
matrix([[1, 2],
        [3, 4]])""")

        def test_pandas(self):
            # noinspection PyPackageRequirements
            import pandas as pd

            df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            self.assert_usual_repr(df)
            self.assert_usual_repr(df.index)
            self.assert_usual_repr(df.a)
            self.assert_usual_repr(df.b)

            df = pd.DataFrame(
                dict((k, range(100)) for k in 'abcdefghijkl')
            ).set_index(['a', 'b'])

            self.assert_cheap_repr(df,
                                   """\
        c   d   e   f  ...   i   j   k   l
a  b                   ...
0  0    0   0   0   0  ...   0   0   0   0
1  1    1   1   1   1  ...   1   1   1   1
2  2    2   2   2   2  ...   2   2   2   2
3  3    3   3   3   3  ...   3   3   3   3
...    ..  ..  ..  ..  ...  ..  ..  ..  ..
96 96  96  96  96  96  ...  96  96  96  96
97 97  97  97  97  97  ...  97  97  97  97
98 98  98  98  98  98  ...  98  98  98  98
99 99  99  99  99  99  ...  99  99  99  99
[100 rows x 10 columns]""")

            self.assert_cheap_repr(df.c,
                                   """\
a   b
0   0      0
1   1      1
2   2      2
3   3      3
          ..
96  96    96
97  97    97
98  98    98
99  99    99
Name: c, Length: 100, dtype: int64""")

            if version_info[:2] < (3, 6):
                self.assert_cheap_repr(df.index,
                                       """\
MultiIndex(levels=[Int64Index(dtype=dtype('int64'), name='a', length=100), Int64Index(dtype=dtype('int64'), name='b', length=100)],
           codes=[FrozenNDArray([ 0,  1,  2, ..., 97, 98, 99], dtype=int8), FrozenNDArray([ 0,  1,  2, ..., 97, 98, 99], dtype=int8)],
           names=['a', 'b'])""")
            else:
                self.assert_cheap_repr(df.index,
                                       """\
MultiIndex(levels=[Int64Index(dtype=dtype('int64'), length=100), Int64Index(dtype=dtype('int64'), length=100)],
           codes=[array([ 0,  1,  2, ..., 97, 98, 99], dtype=int8), array([ 0,  1,  2, ..., 97, 98, 99], dtype=int8)],
           names=['a', 'b'])""")

            values = [4, 2, 3, 1]
            cats = pd.Categorical([1, 2, 3, 4], categories=values)
            self.assert_cheap_repr(
                pd.DataFrame(
                    {"strings": ["a", "b", "c", "d"], "values": values},
                    index=cats).index,
                "CategoricalIndex(categories=Int64Index(dtype=dtype('int64'), "
                "length=4), ordered=False, dtype='category', length=4)"
            )

            if sys.version_info[:2] == (3, 7):
                expected = """\
IntervalIndex(closed='right',
              dtype=interval[int64, right])"""
            else:
                expected = """\
IntervalIndex(closed='right',
              dtype=interval[int64])"""
            self.assert_cheap_repr(pd.interval_range(start=0, end=5), expected)

    def test_bytes(self):
        self.assert_usual_repr(b'')
        self.assert_usual_repr(b'123')
        self.assert_cheap_repr(b'abc' * 50,
                               "b'abcabcabcabcabcabcabcabcabca...bcabcabcabcabcabcabcabcabcabc'"
                               .lstrip('b' if PY2 else ''))

    def test_str(self):
        self.assert_usual_repr('')
        self.assert_usual_repr(u'')
        self.assert_usual_repr(u'123')
        self.assert_usual_repr('123')
        self.assert_cheap_repr('abc' * 50,
                               "'abcabcabcabcabcabcabcabcabca...bcabcabcabcabcabcabcabcabcabc'")

    def test_inheritance(self):
        class A(object):
            def __init__(self):
                pass

        class B(A):
            pass

        class C(A):
            pass

        class D(C):
            pass

        class C2(C):
            pass

        class C3(C, B):
            pass

        class B2(B, C):
            pass

        class A2(A):
            pass

        @register_repr(A)
        def repr_A(_x, _helper):
            return 'A'

        @register_repr(C)
        def repr_C(_x, _helper):
            return 'C'

        @register_repr(B)
        def repr_B(_x, _helper):
            return 'B'

        @register_repr(D)
        def repr_D(_x, _helper):
            return 'D'

        self.assert_cheap_repr(A(), 'A')
        self.assert_cheap_repr(B(), 'B')
        self.assert_cheap_repr(C(), 'C')
        self.assert_cheap_repr(D(), 'D')
        self.assert_cheap_repr(C2(), 'C')
        self.assert_cheap_repr(C3(), 'C')
        self.assert_cheap_repr(B2(), 'B')
        self.assert_cheap_repr(A2(), 'A')

        self.assertEqual(find_repr_function(A), repr_A)
        self.assertEqual(find_repr_function(B), repr_B)
        self.assertEqual(find_repr_function(C), repr_C)
        self.assertEqual(find_repr_function(D), repr_D)
        self.assertEqual(find_repr_function(C2), repr_C)
        self.assertEqual(find_repr_function(C3), repr_C)
        self.assertEqual(find_repr_function(B2), repr_B)
        self.assertEqual(find_repr_function(A2), repr_A)

    def test_exceptions(self):
        with temp_attrs(cheap_repr, 'raise_exceptions', True):
            self.assertRaises(ValueError,
                              lambda: cheap_repr(ErrorClass(True)))

        for C in [ErrorClass, OldStyleErrorClass]:
            name = C.__name__
            self.assert_usual_repr(C())
            warning_message = "Exception 'ValueError' in repr_object for object of type %s. " \
                              "The repr has been suppressed for this type." % name
            if PY2 and C is OldStyleErrorClass:
                warning_message = warning_message.replace('repr_object', 'repr')
            self.assert_cheap_repr_warns(
                C(True),
                warning_message,
                '<%s instance at 0xXXX (exception in repr)>' % name,
            )
            self.assert_cheap_repr(C(), '<%s instance at 0xXXX (repr suppressed)>' % name)
        for C in [ErrorClassChild, OldStyleErrorClassChild]:
            name = C.__name__
            self.assert_cheap_repr(C(), '<%s instance at 0xXXX (repr suppressed)>' % name)

    def test_func_raise_exceptions(self):
        class T(object):
            pass

        @register_repr(T)
        def bad_repr(*_):
            raise TypeError()

        bad_repr.raise_exceptions = True

        self.assertRaises(TypeError, lambda: cheap_repr(T()))

        class X(object):
            def __repr__(self):
                raise IOError()

        class Y:  # old-style in python 2
            def __repr__(self):
                raise IOError()

        raise_exceptions_from_default_repr()

        for C in [X, Y]:
            self.assertRaises(IOError, lambda: cheap_repr(C()))

    def test_default_too_long(self):
        self.assert_usual_repr(DirectRepr('hello'))
        self.assert_cheap_repr_warns(
            DirectRepr('long' * 500),
            'DirectRepr.__repr__ is too long and has been suppressed. '
            'Register a repr for the class to avoid this warning '
            'and see an informative repr again, '
            'or increase cheap_repr.suppression_threshold',
            'longlonglonglonglonglonglong...glonglonglonglonglonglonglong')
        self.assert_cheap_repr(DirectRepr('hello'),
                               '<DirectRepr instance at 0xXXX (repr suppressed)>')

    def test_maxparts(self):
        self.assert_cheap_repr(list(range(8)),
                               '[0, 1, 2, ..., 5, 6, 7]')
        self.assert_cheap_repr(list(range(20)),
                               '[0, 1, 2, ..., 17, 18, 19]')
        with temp_attrs(find_repr_function(list), 'maxparts', 10):
            self.assert_cheap_repr(list(range(8)),
                                   '[0, 1, 2, 3, 4, 5, 6, 7]')
            self.assert_cheap_repr(list(range(20)),
                                   '[0, 1, 2, 3, 4, ..., 15, 16, 17, 18, 19]')

    def test_recursive(self):
        lst = [1, 2, 3]
        lst.append(lst)
        self.assert_cheap_repr(lst, '[1, 2, 3, [1, 2, 3, [1, 2, 3, [...]]]]')

        d = {1: 2, 3: 4}
        d[5] = d
        self.assert_cheap_repr(
            d, '{1: 2, 3: 4, 5: {1: 2, 3: 4, 5: {1: 2, 3: 4, 5: {...}}}}')

    def test_custom_set(self):
        self.assert_cheap_repr(RangeSet(0), 'RangeSet()')
        self.assert_cheap_repr(RangeSet(3), 'RangeSet({0, 1, 2})')
        self.assert_cheap_repr(RangeSet(10), 'RangeSet({0, 1, 2, 3, 4, 5, ...})')

    def test_repr_function(self):
        def some_really_really_long_function_name():
            yield 3

        self.assert_usual_repr(some_really_really_long_function_name, normalise=True)
        self.assert_usual_repr(some_really_really_long_function_name(), normalise=True)
        self.assert_usual_repr(os)

    @skipIf(PY2 and PYPY, "Not supported in pypy2")
    def test_repr_long_class_name(self):
        class some_really_really_long_class_name(object):
            pass

        self.assert_usual_repr(some_really_really_long_class_name(), normalise=True)

    def test_function_names_unique(self):
        # Duplicate function names can lead to mistakes
        assert_unique(f.__name__ for f in set(repr_registry.values()))

    @skipUnless(PY2, "Old style classes only exist in Python 2")
    def test_old_style_class(self):
        self.assert_cheap_repr(OldStyleClass, '<class tests.utils.OldStyleClass at 0xXXX>')

    def test_target_length(self):
        target = 100
        lst = []
        for i in range(100):
            lst.append(i)
            r = cheap_repr(lst, target_length=target)
            usual = repr(lst)
            assert (
                    (r == usual and len(r) < target) ^
                    ('...' in r and len(r) > target)
            )

        self.assertEqual(
            cheap_repr(list(range(100)), target_length=target),
            '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ..., 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]'
        )

        # Don't want to deal with ordering in older pythons
        if version_info[:2] >= (3, 6):
            self.assertEqual(
                cheap_repr(set(range(100)), target_length=target),
                '{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, ...}'
            )

            self.assertEqual(
                cheap_repr({x: x * 2 for x in range(100)}, target_length=target),
                '{0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 12, 7: 14, 8: 16, 9: 18, 10: 20, 11: 22, 12: 24, 13: 26, ...}',
            )


if __name__ == '__main__':
    unittest.main()
