"""
Compilar el motor C de tensores:
    cd src/lib/mathdoz && python3 setup_tensor_c.py build_ext --inplace
"""
from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
    ext = cythonize(
        Extension("tensor_c", ["tensor_c.pyx"]),
        compiler_directives={"language_level": "3"},
    )
except ImportError:
    raise SystemExit("Cython no instalado. Ejecuta: pip install cython")

setup(name="tensor_c", ext_modules=ext)
