#!/usr/bin/env python

from setuptools import setup
import os
import platform

# Run with python setup.py develop --user

# Specify specific compiler on Mac
if platform.system() == 'Darwin':
    os.environ["CC"] = "gcc-mp-5"

setup_requires = []

scripts = []
# scripts.extend(glob('scripts/*py'))


CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)
Operating System :: MacOS
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python
Topic :: Scientific/Engineering

"""
setup(
    name="pool",
    version="0.0.1",
    packages=['pool',
              ],
    #   scripts = [''],
    #
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        'numpy>=1.8',
        'matplotlib>=1.5.3',
        'scipy>=0.19.0',
        'networkx>=2.1',
        'python-louvain',  # Imported as community
        # 'pandas>=0.21.1'
        # 'scikit-image>=0.9.3',
        # 'shapely>=1.2.14',
        # 'scikit-learn>=0.11',
        # 'pillow>=2.6.1',
        # 'future>=0.14',
    ],
    scripts=scripts,
    # package_data={
    #     'replay': [
    #         'tests/*.py',
    #         'tests/data/example.tif',
    #         'tests/data/example.h5',
    #         'tests/data/example-volume.h5',
    #         'tests/data/example-tiffs/*.tif',
    #     ]
    # },
    #
    # metadata for upload to PyPI
    author="Arthur Sugden",
    author_email="arthur.sugden@gmail.com",
    description="Andermann Lab cortical reactivation analysis",
    license="GNU GPLv2",
    keywords="imaging microscopy neuroscience behavior",
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    setup_requires=setup_requires,
    # setup_requires=['setuptools_cython'],
    url="https://www.andermannlab.com/",
    platforms=["Linux", "Mac OS-X", "Windows"],
    ext_modules=[],
    #
    # could also include long_description, download_url, etc.
)