# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import unittest

import torch

from setuptools import setup, Extension
from Cython.Build import cythonize

torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')

extensions = [
    Extension(
        '*', [ "kbc/lib/*.pyx" ],
        language='c++',
        extra_compile_args=[
            "-g", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"
        ],
        include_dirs = [os.path.join(torch_lib, 'include')],
        extra_link_args=[
            "-Wl,-rpath,{}".format(torch_lib),
            os.path.join(torch_lib, 'libcaffe2.so'),
            os.path.join(torch_lib, 'libcaffe2_gpu.so'),
            "-fopenmp"
        ]
    ),
]

setup(
    name = 'kbc',
    include_dirs = [
        np.get_include(), os.getcwd() + '/models/',
    ],
    ext_package = '',
    ext_modules = cythonize(
        extensions,
        language='c++'
    ),
    packages=[
        'kbc',
        'kbc.learning', 'kbc.datasets',
        'kbc.lib',
    ],
    package_data={'kbc': ['data/**/*']},
)
