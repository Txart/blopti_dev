#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 09:07:02 2020

@author: txart
"""

from numpy.distutils.core import Extension

ext1 = Extension(name = 'fd',
                 sources = ['fd1.pyf', 'finite-diff.f95'])

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name = 'f2py_fd',
          description       = "F2PY finite differences",
          author            = "I Urzainki",
          ext_modules = [ext1]
          )

