'''
python setup.py build_ext -i
to compile
'''

from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import os
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)
setup(
    name='Sim3DR_Cython',  # not the package name
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("Sim3DR_Cython",
                           sources=["lib/rasterize.pyx", "lib/rasterize_kernel.cpp"],
                           language='c++',
                           include_dirs=[numpy.get_include()],
                           extra_compile_args=["-std=c++11"])],
)
