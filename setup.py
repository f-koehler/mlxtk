#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from distutils.core import setup
from setuptools import setup

setup(
    name="mlxtk",
    version="0.0.0",
    author="Fabian Köhler",
    author_email="fkoehler@physnet.uni-hamburg.de",
    url="https://github.com/f-koehler/mlxtk",
    license="MIT",
    install_requires=[
        "colorama==0.3.9",
        "h5py==2.7.1",
        "matplotlib==2.1.0",
        "numpy==1.13.3",
        "pandas==0.21.0",
        "scipy==1.0.0",
    ],
    packages=["mlxtk"],
    entry_points={
        "console_scripts": [
            "plot_energy=mlxtk.scripts.plot_energy:main",
            "plot_gpop=mlxtk.scripts.plot_gpop:main",
            "plot_gpop_slider=mlxtk.scripts.plot_gpop_slider:main",
            "plot_gpop3d=mlxtk.scripts.plot_gpop3d:main",
            "plot_natpop=mlxtk.scripts.plot_natpop:main",
            "plot_norm=mlxtk.scripts.plot_norm:main",
            "plot_overlap=mlxtk.scripts.plot_overlap:main",
            # "plot_dmat1=mlxtk.scripts.plot_dmat1:main",
            # "plot_dmat2=mlxtk.scripts.plot_dmat2:main",
            "scan_view=mlxtk.scripts.scan_view:main",
        ]
    })
