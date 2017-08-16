#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup

setup(
    name="mlxtk",
    version="0.0.0",
    author="Fabian Köhler",
    author_email="fkoehler@physnet.uni-hamburg.de",
    url="https://github.com/f-koehler/mlxtk",
    packages=["mlxtk"],
    entry_points={
        "console_scripts": [
            "plot_dmat1=scripts.plot_dmat1:main",
            "plot_dmat2=scripts.plot_dmat2:main",
            "plot_energy=scripts.plot_energy:main",
            "plot_natpop=scripts.plot_natpop:main",
            "plot_norm=scripts.plot_norm:main",
        ]
    })
