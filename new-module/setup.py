from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension("self_learning_cython", sources=["self_learning_cython.pyx"], libraries=["m"],
              include_dirs=[np.get_include()], extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp'], reload_support=True)
]

setup(
    name="probabilistic_classifier_cython", cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(extensions, language_level="3"),
)