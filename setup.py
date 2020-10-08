#! /usr/bin/env python

from distutils.core import setup

from overload import __version__, __doc__

# perform the setup action
setup(
    name = "overload",
    version = __version__,
    description = "simple provision of method overloading",
    long_description = __doc__,
    author = "Richard Jones",
    author_email = "richard@python.org",
    py_modules = ['overload'],
    url = 'http://pypi.python.org/pypi/overload',
    classifiers = [
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Code Generators',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing :: Markup :: HTML',
        'License :: OSI Approved :: BSD License',
    ],
)

# vim: set filetype=python ts=4 sw=4 et si
