import traceback

from qualname import qualname


def safe_qualname(cls):
    # type: (type) -> str
    result = _safe_qualname_cache.get(cls)
    if not result:
        try:
            result = qualname(cls)
        except (AttributeError, IOError, SyntaxError):
            result = cls.__name__
        if '<locals>' not in result:
            _safe_qualname_cache[cls] = result
    return result


_safe_qualname_cache = {}


def type_name(x):
    return safe_qualname(x.__class__)


def exception_string(exc):
    assert isinstance(exc, BaseException)
    return ''.join(traceback.format_exception_only(type(exc), exc)).strip()
