from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "BDiF Data Scrubber",
    ext_modules = cythonize(['data_handler.pyx', 'scrub_handler.pyx', 'utils.pyx', 'stats_handler.pyx']),  # accepts a glob pattern
)

