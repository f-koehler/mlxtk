"""Hashing functions.

Functions to hash files and other data using standard hashing programs such
as ``sha256sum``.
"""
import subprocess

import numpy


def hash_string(data: str, program: str = "sha1sum"):
    return subprocess.check_output([program],
                                   input=data.encode()).decode().split()[0]


def inaccurate_hash(x: numpy.ndarray, decimals: int = 10, program="sha256sum"):
    return (subprocess.check_output(
        [program], input=x.round(decimals).tobytes("C")).decode().split()[0])
