from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "Large dataset parser functions",
    ext_modules = cythonize('parser.pyx'),  # accepts a glob pattern
)

