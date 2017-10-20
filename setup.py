from distutils.core import setup
from distutils.extension import Extension
# from Cython.Distutils import build_ext
import numpy as np
import os
import re
import sys
import tensorflow as tf

tf_include = tf.sysconfig.get_include()

# gcc 4 or using already-built binary,then set USE_CXX11_ABI=1
USE_CXX11_ABI = 0


def find_packages(path):
    ret = []
    for root, dirs, files in os.walk(path):
        if '__init__.py' in files:
            ret.append(re.sub('^[^A-z0-9_]+', '', root.replace('/', '.')))
    return ret


try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


platform_args = []

if sys.platform == "darwin":
    platform_args.append("-undefined dynamic_lookup")

# ext_modules
matpackops = Extension(
    "autogp.util.matpackops",
    [
        "autogp/util/tf_ops/vec_to_tri.cc",
        "autogp/util/tf_ops/tri_to_vec.cc"
    ],
    language='c++',  # Needed?
    extra_compile_args=[
        '-Wno-cpp',
        "-Wno-unused-function",
        "-std=c++11",  # Needed?
        "-shared"  # Needed?
    ] + platform_args,
    include_dirs=[tf_include]
    )

ext_modules = [matpackops]

REQUIRED = [
    'scikit-learn>=0.17.0',
    'tensorflow>=1.1.0',
]

# class custom_build_ext(build_ext):
#     def build_extensions(self):
#         customize_compiler(self.compiler)
#         build_ext.build_extensions(self)


setup(
    name='AutoGP',
    version='0.1',
    description='Unified tool for automatric Gaussian Process Inference',
    author='Karl Krauth and Edwin Bonilla',
    author_email='edwinbonilla+autogp@gmail.com',
    url='https://github.com/ebonilla/AutoGP',
    license='Apache',
    packages=find_packages('autogp'),
    ext_modules=ext_modules,
    install_requires=REQUIRED,
    cmdclass={
        # 'build_ext': custom_build_ext
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
