import collections
import functools
import importlib.util
import itertools
import os.path
import shutil
import signal
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Iterable, List, Union

import numba
import numpy
import tqdm
from pathos.pools import ProcessPool as Pool

from .log import get_logger

LOGGER = get_logger(__name__)


def copy_file(src: Union[str, Path], dst: Union[str, Path]):
    src = str(src)
    dst = str(dst)
    LOGGER.debug("copy file: %s -> %s", src, dst)
    shutil.copy2(src, dst)


def get_common_path(paths: List[Path]) -> Path:
    return Path(os.path.commonpath([str(p) for p in paths]))


def make_path(path: Union[str, Path]) -> Path:
    if isinstance(path, Path):
        return path
    return Path(path)


def labels_from_paths(paths: List[Path]) -> List[str]:
    common = get_common_path(paths)
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

        value = self.func(*args)
        self.cache[args] = value
        return value

    def __repr__(self):
        return self.func.__doc__

    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)


def load_module(path: str):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@numba.vectorize(
    [numba.float64(numba.complex128),
     numba.float32(numba.complex64)])
def compute_magnitude(x):
    return x.real**2 + x.imag**2


@numba.vectorize([
    numba.float64(numba.float64, numba.float64),
    numba.float32(numba.float32, numba.float32)
])
def compute_magnitude_split(real, imag):
    return real**2 + imag**2


def round_robin(*iterables) -> Iterable[Any]:
    num_non_empty = len(iterables)
    cycles = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_non_empty:
        try:
            for cycle in cycles:
                yield cycle()
        except StopIteration:
            num_non_empty -= 1
            cycles = itertools.cycle(itertools.islice(cycles, num_non_empty))


def map_parallel_progress(func, items: List[Any],
                          processes: int = cpu_count()):
    with Pool(processes=processes) as pool:
        with tqdm.tqdm(total=len(items)) as pbar:
            results = []
            for result in pool.imap(func, items):
                pbar.update()
                results.append(result)
            return results
