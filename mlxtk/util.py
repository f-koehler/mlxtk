import collections
import functools
import os.path
from pathlib import Path
from typing import List

import numpy


def labels_from_paths(paths: List[str]) -> List[str]:
    common = os.path.commonpath(paths)
    parts = [Path(os.path.relpath(p, common)).parts for p in paths]
    arr = numpy.full((len(parts), len(max(parts, key=len))), "", dtype=object)
    for i, _ in enumerate(parts):
        for j, _ in enumerate(parts[i]):
            arr[i][j] = parts[i][j]
    mask = [
        i for i in range(arr.shape[1]) if numpy.unique(arr[:, i]).shape[0] != 1
    ]
    arr = arr[:, mask]
    return ["/".join(a) for a in arr]


class memoize:
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        return self.func.__doc__

    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)
