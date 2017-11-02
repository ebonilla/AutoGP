from distutils.core import setup
import os
import re


def find_packages(path):
    ret = []
    for root, dirs, files in os.walk(path):
        if '__init__.py' in files:
            ret.append(re.sub('^[^A-z0-9_]+', '', root.replace('/', '.')))
    return ret


REQUIRED = [
    'scikit-learn>=0.17.0',
    'tensorflow>=1.1.0',
]


setup(
    name='AutoGP',
    version='0.1',
    description='Unified tool for automatric Gaussian Process Inference',
    author='Karl Krauth and Edwin Bonilla',
    author_email='edwinbonilla+autogp@gmail.com',
    url='https://github.com/ebonilla/AutoGP',
    license='Apache',
    packages=find_packages('autogp'),
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
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
