import os
import sys

# Compile the TensorFlow ops.
compile_command = ("g++ -std=c++11 -shared ./autogp/util/tf_ops/vec_to_tri.cc "
                   "./autogp/util/tf_ops/tri_to_vec.cc -o ./autogp/util/tf_ops/matpackops.so "
                   "-fPIC -I $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')")

if sys.platform == "darwin":
    compile_command += " -undefined dynamic_lookup"

os.system(compile_command)
setup(
    name='AutoGP',
    version='0.1',
    description='Unified tool for automatric Gaussian PRocess Inference',
    author='Karl Krauth and Edwin Bonilla',
    author_email='edwinbonilla+autogp@gmail.com',
    url='https://github.com/ebonilla/AutoGP',
    license='Apache',
    packages=find_packages('autogp'),
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
