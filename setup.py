#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

setup(
    name="mlxtk",
    version="0.0.0",
    author="Fabian Köhler",
    author_email="fkoehler@physnet.uni-hamburg.de",
    url="https://github.com/f-koehler/mlxtk",
    license="MIT",
    install_requires=[
        "colorama",
        "doit",
        "h5py",
        "matplotlib",
        "numpy",
        "numpy-stl",
        "pandas",
        "scipy",
        "tabulate",
    ],
    packages=["mlxtk"],
    entry_points={
        "console_scripts":[
            "gpop_model = mlxtk.scripts.gpop_model:main",
            "plot_energy = mlxtk.scripts.plot_energy:main",
            "plot_gpop = mlxtk.scripts.plot_gpop:main",
            "plot_gpop_slider = mlxtk.scripts.plot_gpop_slider:main",
            "plot_natpop = mlxtk.scripts.plot_natpop:main",
            "spectrum_1b = mlxtk.scripts.spectrum_1b:main",
            ]
        }
)
