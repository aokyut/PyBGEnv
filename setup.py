from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため

# ext = Extension("sample", sources=["sample.pyx"], include_dirs=['.', get_include()])
# setup(name="sample", ext_modules=cythonize([ext]))

ox = Extension("PyBGEnv.ox", sources=["ox/ox.pyx"], include_dirs=['.', get_include()])
qubic = Extension("PyBGEnv.qubic", sources=["qubic/qubic.pyx"], include_dirs=['.', get_include()])
gomoku = Extension("PyBGEnv.gomoku", sources=["gomoku/gomoku.pyx"], include_dirs=['.', get_include()])
connect4 = Extension("PyBGEnv.connect4", sources=["connect4/connect4.pyx"], include_dirs=['.', get_include()])

setup(name="PyBGEnv",
    version="0.0.1",
    ext_modules=cythonize([ox, qubic, connect4]))