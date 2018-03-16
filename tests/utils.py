import warnings
from contextlib import contextmanager
from unittest import TestCase

try:
    from collections import Counter
except ImportError:
    from counter import Counter

try:
    from unittest import skip, skipUnless
except ImportError:
    def skip(_f):
        return lambda self: None


    def skipUnless(condition, _reason):
        if condition:
            return lambda x: x
        else:
            return lambda x: None


@contextmanager
def temp_attrs(*attrs):
    if len(attrs) == 3 and not isinstance(attrs[1], tuple):
        attrs = (attrs,)
    previous = []
    try:
        for obj, attr, val in attrs:
            old = getattr(obj, attr)
            previous.append((obj, attr, old))
            setattr(obj, attr, val)
        yield
    finally:
        for t in previous:
            setattr(*t)


class TestCaseWithUtils(TestCase):
    def assert_warns(self, category, message, f):
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')
            result = f()
            self.assertEqual(len(warning_list), 1)
            self.assertEqual(warning_list[0].category, category)
            self.assertEqual(str(warning_list[0].message), message)
            return result


def assert_unique(items):
    counts = Counter(items)
    dups = [k for k, v in counts.items()
            if v > 1]
    if dups:
        raise ValueError('Duplicates: %s' % dups)


class OldStyleClass:
    pass
