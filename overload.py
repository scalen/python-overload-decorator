'''Simple overloading of functions through an @overload decorator.

This module allows one to provide multiple interfaces for a functions,
methods, classmethods, staticmethods or classes. See below for some notes
about overloading classes, you strange person you.

The appropriate implementation is chosen based on the calling argument
pattern.

For example:

>>> class A(object):
...   @overload
...   def method(self, a):
...     return 'a'
...   @method.add
...   def method(self, a, b):
...     return 'a, b'
... 
>>> a = A()
>>> a.method(1)
'a'
>>> a.method(1, 2)
'a, b'

The overloading handles fixed, keyword, variable (``*args``) and arbitrary
keyword (``**keywords``) arguments.

It also handles annotations if those annotations are types:

>>> @overload
... def func(a:int):
...   return 'int'
... 
>>> @func.add
... def func(a:str):
...   return 'str'
... 
>>> func(1)
'int'
>>> func('s')
'str'
>>> func(1.0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "overload.py", line 94, in f
    raise TypeError('invalid call argument(s)')
TypeError: invalid call argument(s)

This feature (and currently the module in general) requires Python 3.

The docstring and name (ie. documentation) of the resultant callable will
match that of the *first* callable overloaded.


Overloading Classes
--------------------

Overloading classes allows you to select a class type based on the
construction arguments of each alternative type's __new__ method.

There's a catch though: the __new__ method must *explicitly* invoke the
base class __new__ method, rather than use super() like usual. This is
because after being @overloaded the class is a function, and super()
doesn't like being passed functions. So instead of::

    @overload
    class A(object):
        def __new__(cls):
            # this will fail because "A" is a function now
            return super(A, cls).__new__(cls)

you must::

    @overload
    class A(object):
        def __new__(cls):
            # must explicitly reference the base class
            return object.__new__(cls)

I'll leave it up to the reader to justify their use of @overloading
classes.


Version History (in Brief)
--------------------------

- 1.4.0 Support overloading classes with __init__ methods but no __new__
- 1.3.1 Improve type annotation support with stringed types and forward
        references.
- 1.3.0 Add support for positional- and keyword-only arguments.
- 1.2.2 Grand refactor to separate the different kinds of logic:
        * Determining the kind of callable;
        * Parsing callable signature into readable/useable variables;
        * Validating arguments against a signature;
        * Multiple dispatch over multiple signatures.
- 1.2.1 Fixed the case where the first defaulted arg is specified, but the
        second is not (previously used the first default value).
- 1.2.0 Simplified `f()` by distinguishing between definition function and
        implementation to call at decoration time.
- 1.1.0 altered the text of the invalid call TypeError. Removed debug prints.
- 1.0.0 the initial release

See the end of the source file for the license of use.
'''
__version__ = '1.1'

import functools
import types
import unittest
from typing import get_type_hints, Any, Callable, List, Sequence, Tuple, Union


class _Undefined:
    def __bool__(self):
        return False
    def __nonzero__(self):
        return False


_ParamAnnotatedType = Union[type, _Undefined]
_PositionalParamDefinition = Tuple[int, str, _ParamAnnotatedType, Any]
_KeywordParamDefinition = Tuple[str, _ParamAnnotatedType, Any]


class Signature:
    '''Wraps a callable to directly expose its signature.'''
    undefined = _Undefined()

    def __init__(self, f: Callable, /):
        self._callable: Callable = f
        self.implementation: Callable = f
        self._skip_first_arg: bool = isinstance(f, classmethod)

        # if the callable is a class then look for __new__ variations
        if isinstance(f, type):
            # sanity check
            if isinstance(f.__new__, types.FunctionType):
                self._callable = f.__new__
            elif isinstance(f.__init__, types.FunctionType):
                self._callable = f.__init__
            else:
                raise TypeError('overloaded class requires __new__ or __init__ implementation')
            self._skip_first_arg = True
        elif isinstance(f, classmethod) or isinstance(f, staticmethod):
            # actually call the method underlying the classmethod directly - the proxyish
            # object we've received as f has the class first param filled in...
            self._callable = self.implementation = f.__get__(f.__class__)

    def _get_param_type(self, param: str) -> _ParamAnnotatedType:
        annotations = get_type_hints(self._callable)
        if param not in annotations:
            return self.undefined
        annotation = annotations[param]
        return annotation if isinstance(annotation, type) else self.undefined

    def _get_positional_default(self, position: int) -> Any:
        if not self._callable.__defaults__:
            return self.undefined
        default_count = len(self._callable.__defaults__)
        index = position - self._callable.__code__.co_argcount + default_count
        if 0 <= index < default_count:
            return self._callable.__defaults__[index]
        else:
            return self.undefined

    def _get_keyword_default(self, parameter: str) -> Any:
        keyword_only_defaults = getattr(self._callable, "__kwdefaults__", None) or {}
        if parameter in keyword_only_defaults:
            return keyword_only_defaults[parameter]
        else:
            return self.undefined

    @property
    def _positional_only_parameters_count(self) -> int:
        return getattr(self._callable.__code__, "co_posonlyargcount", None) or 0

    @property
    def _positional_parameters(self) -> Sequence[_PositionalParamDefinition]:
        # unlike instance methods, class methods don't appear to have the class passed in as the
        # first arg, so skip filling the first argument
        start = 1 if self._skip_first_arg else 0
        positional_parameters_slice = slice(start, self._callable.__code__.co_argcount)
        return tuple(
            (n, param, self._get_param_type(param), self._get_positional_default(n))
            for n, param in enumerate(
                self._callable.__code__.co_varnames[positional_parameters_slice], start=start
            )
        )

    @property
    def _keyword_only_parameters(self) -> Sequence[_KeywordParamDefinition]:
        keyword_only_parameter_count = (
            getattr(self._callable.__code__, "co_kwonlyargcount", None) or 0
        )
        if not keyword_only_parameter_count:
            return ()

        keyword_only_parameters_slice = slice(
            self._callable.__code__.co_argcount,
            self._callable.__code__.co_argcount + keyword_only_parameter_count,
        )
        return tuple(
            (param, self._get_param_type(param), self._get_keyword_default(param))
            for param in self._callable.__code__.co_varnames[keyword_only_parameters_slice]
        )

    def validate(self, *args, **kwargs) -> bool:
        # duplicate the arguments provided so we may consume them
        _args = list(args)
        _kw = dict(kwargs)

        # validate args (and kwargs, where specified for positional parameters
        for index, param, param_type, default in self._positional_parameters:
            if _args:
                if param in _kw:
                    # Arg specified in both args and kwargs
                    return False
                value = _args.pop(0)
            elif param in _kw:
                # Arg provided as kwarg
                if index < self._positional_only_parameters_count:
                    # Positional-only arg specified in kwargs
                    return False
                value = _kw.pop(param)
            elif default is self.undefined:
                # No value for non-defaulted arg
                return False
            else:
                # Arg not provided, but will be defaulted when called
                # Default should not be type-checked
                continue

            if param_type is not self.undefined and not isinstance(value, param_type):
                # Arg is not of expected type
                return False

        # validate remaining kwargs against keyword-only arguments
        for param, param_type, default in self._keyword_only_parameters:
            if param in _kw:
                value = _kw.pop(param)
                if param_type is not self.undefined and not isinstance(value, param_type):
                    # Keyword-only arg is not of expected type
                    return False
            elif default is self.undefined:
                # No value for non-defaulted keyword-only arg
                return False

        # validate remaining varargs/-kwargs
        vararg_index = -1
        to_verify: List[Tuple[str, Iterable]] = []
        if self._callable.__code__.co_flags & 0x08:
            to_verify.append((self._callable.__code__.co_varnames[vararg_index], kwargs.values()))
            vararg_index = -2
        elif _kw:
            # Varkwargs where none expected
            return False
        if self._callable.__code__.co_flags & 0x04:
            to_verify.append((self._callable.__code__.co_varnames[vararg_index], args))
        elif _args:
            # Varargs where none expected
            return False
        for vararg_name, values in to_verify:
            param_type = self._get_param_type(vararg_name)
            if param_type is not self.undefined and not all(
                isinstance(v, param_type) for v in values
            ):
                # Varargs/-kwargs of unexpected types are present
                return False

        return True


def overload(callable):
    '''Allow overloading of a callable.

    Invoke the result of this call with .add() to add additional
    implementations.
    '''
    definitions: List[Signature] = []

    @functools.wraps(callable)
    def multiple_dispatch(*args, **kwargs):
        for definition in definitions:
            if definition.validate(*args, **kwargs):
                # attempt to invoke the callable
                try:
                    return definition.implementation(*args, **kwargs)
                except (TypeError, ValueError) as e:
                    continue

        # this error message probably can't get any better :-)
        raise TypeError('invalid call argument(s)')

    # allow adding of additional implementations
    def add(callable):
        definitions.append(Signature(callable))
        return multiple_dispatch
    multiple_dispatch.add = add

    multiple_dispatch.add(callable)
    return multiple_dispatch


class TestOverload(unittest.TestCase):
    def test_wrapping(self):
        'check that we generate a nicely-wrapped result'
        @overload
        def func(arg):
            'doc'
            pass
        @func.add
        def func(*args):
            'doc2'
            pass
        self.assertEqual(func.__doc__, 'doc')

    def test_method(self):
        'check we can overload instance methods'
        class A:
            @overload
            def method(self):
                return 'ok'
            @method.add
            def method(self, *args):
                return 'args'
        self.assertEqual(A().method(), 'ok')
        self.assertEqual(A().method(1), 'args')

    def test_classmethod(self):
        'check we can overload classmethods'
        class A:
            @overload
            @classmethod
            def method(cls):
                return 'ok'
            @method.add
            @classmethod
            def method(cls, *args):
                return 'args'
        self.assertEqual(A.method(), 'ok')
        self.assertEqual(A.method(1), 'args')

    def test_staticmethod(self):
        'check we can overload staticmethods'
        class A:
            @overload
            @staticmethod
            def method():
                return 'ok'
            @method.add
            @staticmethod
            def method(*args):
                return 'args'
        self.assertEqual(A.method(), 'ok')
        self.assertEqual(A.method(1), 'args')

    def test_class(self):
        @overload
        class A(object):
            first = True
            def __new__(cls):
                # must explicitly reference the base class
                return object.__new__(cls)

        @A.add
        class A(object):
            first = False
            def __new__(cls, a):
                # must explicitly reference the base class
                return object.__new__(cls)
        self.assertEqual(A().first, True)
        self.assertEqual(A(1).first, False)

    def test_arg_pattern(self):
        @overload
        def func(a):
            return 'with a'

        @func.add
        def func(a, b):
            return 'with a and b'

        self.assertEqual(func('a'), 'with a')
        self.assertEqual(func('a', 'b'), 'with a and b')
        self.assertRaises(TypeError, func)
        self.assertRaises(TypeError, func, 'a', 'b', 'c')
        self.assertRaises(TypeError, func, b=1)

    def test_positional_only_args(self):
        @overload
        def func(a, /, b):
            return 'with b and positional-only a'

        @func.add
        def func(a, b):
            return 'with a and b'

        self.assertEqual(func('a', b='b'), 'with b and positional-only a')
        self.assertEqual(func(a='a', b='b'), 'with a and b')

    def test_overload_independent(self):
        class A(object):
            @overload
            def method(self):
                return 'a'

        class B(object):
            @overload
            def method(self):
                return 'b'

        self.assertEqual(A().method(), 'a')
        self.assertEqual(B().method(), 'b')

    def test_arg_types(self):
        @overload
        def func(a:int):
            return 'int'

        @func.add
        def func(a:str):
            return 'str'

        @func.add
        def func(a:Union[dict, list]):
            return 'dict or list'

        self.assertEqual(func(1), 'int')
        self.assertEqual(func('1'), 'str')
        self.assertEqual(func({}), 'dict or list')

    def test_varargs(self):
        @overload
        def func(a):
            return 'a'

        @func.add
        def func(*args):
            return '*args {}'.format(len(args))

        self.assertEqual(func(1), 'a')
        self.assertEqual(func(1, 2), '*args 2')

    def test_varargs_types(self):
        @overload
        def func(*args: int):
            return 'int'

        @func.add
        def func(*args: str):
            return 'str'

        @func.add
        def func(*args):
            return 'any'

        self.assertEqual(func(1), 'int')
        self.assertEqual(func('1', '2'), 'str')
        self.assertEqual(func(1, '2'), 'any')

    def test_varargs_mixed(self):
        @overload
        def func(a):
            return 'a'

        @func.add
        def func(a, *args):
            return '*args {}'.format(len(args))

        self.assertEqual(func(1), 'a')
        self.assertEqual(func(1, 2), '*args 1')
        self.assertEqual(func(1, 2, 3), '*args 2')

    def test_kw(self):
        @overload
        def func(a):
            return 'a'

        @func.add
        def func(**kw):
            return '**kw {}'.format(len(kw))

        self.assertEqual(func(1), 'a')
        self.assertEqual(func(a=1), 'a')
        self.assertEqual(func(a=1, b=2), '**kw 2')

    def test_kw_only_args(self):
        @overload
        def func(a, *, b):
            return 'with a and keyword only b'

        @func.add
        def func(a, b):
            return 'with a and b'

        self.assertEqual(func(1, 2), 'with a and b')
        self.assertEqual(func(a=1, b=2), 'with a and keyword only b')

    def test_kw_types(self):
        @overload
        def func(**kw: int):
            return 'int'

        @func.add
        def func(**kw: str):
            return 'str'

        @func.add
        def func(**kw):
            return 'any'

        self.assertEqual(func(a=1), 'int')
        self.assertEqual(func(b='1', a='2'), 'str')
        self.assertEqual(func(b=1, a='2'), 'any')

    def test_kw_mixed(self):
        @overload
        def func(a):
            return 'a'

        @func.add
        def func(a, **kw):
            return '**kw {}'.format(len(kw))

        self.assertEqual(func(1), 'a')
        self.assertEqual(func(a=1), 'a')
        self.assertEqual(func(a=1, b=2), '**kw 1')

    def test_kw_mixed2(self):
        @overload
        def func(a):
            return 'a'

        @func.add
        def func(c=1, **kw):
            return '**kw {}'.format(len(kw))

        self.assertEqual(func(1), 'a')
        self.assertEqual(func(a=1), 'a')
        self.assertEqual(func(c=1, a=2), '**kw 1')

    def test_kw_mixed3(self):
        @overload
        def func(a):
            return 'a'

        @func.add
        def func(a=1, b=2, c=3, **kw):
            return 'a {a}, b {b}, c {c}, **kw {count}'.format(a=a, b=b, c=c, count=len(kw))

        self.assertEqual(func(1), 'a')
        self.assertEqual(func(a=1), 'a')
        self.assertEqual(func(a=4, c=5, d=0), 'a 4, b 2, c 5, **kw 1')

if __name__ == '__main__':
    unittest.main()


# Copyright (c) 2011 Richard Jones <richard@mechanicalcat.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# vim: set filetype=python ts=4 sw=4 et si
