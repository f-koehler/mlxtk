#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

setup(
    name="mlxtk",
    version="0.3.1",
    author="Fabian Köhler",
    author_email="fkoehler@physnet.uni-hamburg.de",
    description="Toolkit to design, run and analyze ML-MCTDH(X) simulations",
    long_description="""
mlxtk gives the user a simple interface to setup physical systems and provides
common simulation tasks to be used as building blocks to set up rather complex
simulations. Data is automatically stored in efficient formats (i.e. HDF5 and
gzipped files).

Simulations can also be used in the context of parameter scans where each
simulation is executed for each specified parameter combination. Submission
of simulation jobs to computing clusters is easily achieved from the command
line.

Furthermore, analysis and plotting tools are provided to interpret the
simulation outcome.
""",
    url="https://github.com/f-koehler/mlxtk",
    license="MIT",
    install_requires=[
        "colorama",
        "doit",
        "h5py",
        "jinja2",
        "matplotlib",
        "numpy",
        "numpy-stl",
        "pandas",
        "scipy",
        "tabulate",
    ],
    packages=["mlxtk"],
    entry_points={
        "console_scripts": [
            "gpop_model = mlxtk.scripts.gpop_model:main",
            "plot_energy = mlxtk.scripts.plot_energy:main",
            "plot_energy_diff = mlxtk.scripts.plot_energy_diff:main",
            "plot_entropy = mlxtk.scripts.plot_entropy:main",
            "plot_entropy_diff = mlxtk.scripts.plot_entropy_diff:main",
            "plot_expval = mlxtk.scripts.plot_expval:main",
            "plot_gpop = mlxtk.scripts.plot_gpop:main",
            "plot_natpop = mlxtk.scripts.plot_natpop:main",
            "spectrum_1b = mlxtk.scripts.spectrum_1b:main",
        ]
    },
    keywords=["physics quantum dynamics ML-MCTDHX MCTDHB MCTDHF dataflow"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Environment :: MacOS X",
        "Environment :: X11 Applications :: Qt",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.5",
    zip_safe=True, )
