#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(name='guardrail',
      version='1.0',
      description='guardrail: A safety classifier evaluation toolkit.',
      author='Allen Institute for AI',
      python_requires=">=3.10",
      packages=find_packages(),
      )
