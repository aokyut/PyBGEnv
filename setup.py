from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため

# ext = Extension("sample", sources=["sample.pyx"], include_dirs=['.', get_include()])
# setup(name="sample", ext_modules=cythonize([ext]))

ox = Extension("PyBGEnv.ox", sources=["ox/ox.pyx"], include_dirs=['.', get_include()])
qubic = Extension("PyBGEnv.qubic", sources=["qubic/qubic.pyx"], include_dirs=['.', get_include()])
setup(name="PyBGEnv",
    ext_modules=cythonize([ox, qubic]))