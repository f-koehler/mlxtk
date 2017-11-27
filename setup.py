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
            "plot_energy=mlxtk.scripts.plot_energy:main",
            "plot_gpop=mlxtk.scripts.plot_gpop:main",
            "plot_gpop3d=mlxtk.scripts.plot_gpop3d:main",
            "plot_natpop=mlxtk.scripts.plot_natpop:main",
            "plot_norm=mlxtk.scripts.plot_norm:main",
            "plot_overlap=mlxtk.scripts.plot_overlap:main",
            # "plot_dmat1=mlxtk.scripts.plot_dmat1:main",
            # "plot_dmat2=mlxtk.scripts.plot_dmat2:main",
            "scan_view=mlxtk.scripts.scan_view:main",
        ]
    })
