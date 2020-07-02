import collections
import functools
import importlib.util
import itertools
import os.path
import re
import shutil
import subprocess
import sys
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

# import numba
import numpy
import tqdm
from pathos.pools import ProcessPool as Pool

from mlxtk import cwd
from mlxtk.log import get_logger

LOGGER = get_logger(__name__)


def copy_file(src: Union[str, Path], dst: Union[str, Path]):
    src = str(src)
    dst = str(dst)
    LOGGER.debug("copy file: %s -> %s", src, dst)
    shutil.copy2(src, dst)


def remove_file(path: Union[str, Path]):
    path = make_path(path)
    if path.exists() and path.is_file():
        LOGGER.debug("delete file: %s", str(path))
        path.unlink()
    else:
        LOGGER.debug("file does not exist/path is not a file, do nothing")


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
    mask = [i for i in range(arr.shape[1]) if numpy.unique(arr[:, i]).shape[0] != 1]
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


# @numba.vectorize(
#     [numba.float64(numba.complex128),
#      numba.float32(numba.complex64)])
def compute_magnitude(x):
    return x.real ** 2 + x.imag ** 2


# @numba.vectorize([
#     numba.float64(numba.float64, numba.float64),
#     numba.float32(numba.float32, numba.float32)
# ])
def compute_magnitude_split(real, imag):
    return real ** 2 + imag ** 2


REGEX_FOLDER_SIZE = re.compile(r"^(\d+)\s+")


def get_folder_size(path: Union[str, Path]) -> int:
    path = str(path)
    m = REGEX_FOLDER_SIZE.match(
        subprocess.check_output(["du", "-s", "-b", path]).decode()
    )
    if not m:
        raise RuntimeError('Error getting folder size for "{}"'.format(path))
    return int(m.group(1))


def compress_folder(path: Union[str, Path], compression: int = 9, jobs: int = 1):
    path = make_path(path)
    folder = path.name
    archive = path.with_suffix(".tar.gz").name

    exe_pv = shutil.which("pv")
    exe_tar = shutil.which("tar")
    exe_gzip = shutil.which("tar")
    exe_pigz = shutil.which("pigz")

    with cwd.WorkingDir(path.parent):
        if exe_pigz:
            if exe_pv:
                with open(archive, "wb") as fptr:
                    size = get_folder_size(folder)
                    process_tar = subprocess.Popen(
                        [exe_tar, "cf", "-", folder], stdout=subprocess.PIPE
                    )
                    process_pv = subprocess.Popen(
                        [exe_pv, "-s", str(size)],
                        stdin=process_tar.stdout,
                        stdout=subprocess.PIPE,
                    )
                    process_pigz = subprocess.Popen(
                        [exe_pigz, "-" + str(compression), "-p", str(jobs)],
                        stdin=process_pv.stdout,
                        stdout=fptr,
                    )
                    process_tar.wait()
                    process_pv.wait()
                    process_pigz.wait()
            else:
                LOGGER.warning("cannot find pv, no progress will be displayed")
                with open(archive, "wb") as fptr:
                    process_tar = subprocess.Popen(
                        [exe_tar, "cf", "-", folder], stdout=subprocess.PIPE
                    )
                    process_pigz = subprocess.Popen(
                        [exe_pigz, "-" + str(compression), "-p", str(jobs)],
                        stdin=process_tar.stdout,
                        stdout=fptr,
                    )
                    process_tar.wait()
                    process_pigz.wait()
        elif exe_gzip:
            if jobs > 1:
                LOGGER.warning(
                    "gzip does not support parallel compression, using one thread only"
                )
            if exe_pv:
                with open(archive, "wb") as fptr:
                    size = get_folder_size(folder)
                    process_tar = subprocess.Popen(
                        [exe_tar, "cf", "-", folder], stdout=subprocess.PIPE
                    )
                    process_pv = subprocess.Popen(
                        [exe_pv, "-s", str(size)],
                        stdin=process_tar.stdout,
                        stdout=subprocess.PIPE,
                    )
                    process_gzip = subprocess.Popen(
                        [exe_gzip, "-" + str(compression)],
                        stdin=process_pv.stdout,
                        stdout=fptr,
                    )
                    process_tar.wait()
                    process_pv.wait()
                    process_gzip.wait()
            else:
                LOGGER.warning("cannot find pv, no progress will be displayed")
                with open(archive, "wb") as fptr:
                    process_tar = subprocess.Popen(
                        [exe_tar, "cf", "-", folder], stdout=subprocess.PIPE
                    )
                    process_gzip = subprocess.Popen(
                        [exe_gzip, "-" + str(compression)],
                        stdin=process_gzip.stdout,
                        stdout=fptr,
                    )
                    process_tar.wait()
                    process_gzip.wait()
        else:
            raise RuntimeError("Cannot find either pigz or gzip")


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


def map_parallel_progress(func, items: List[Any], processes: int = cpu_count()):
    with Pool(processes=processes) as pool:
        with tqdm.tqdm(total=len(items)) as pbar:
            results = []
            for result in pool.imap(func, items):
                pbar.update()
                results.append(result)
            return results


def list_files(
    path: Union[Path, str], extensions: Optional[List[str]] = None
) -> List[Path]:
    path = make_path(path)
    files = sorted(p for p in path.iterdir() if p.is_file())
    if not extensions:
        return list(files)

    return [p for p in files if p.suffix in extensions]


def list_dirs(path: Union[Path, str]) -> List[Path]:
    path = make_path(path)
    return list(sorted(p for p in path.iterdir() if p.is_dir()))


def create_relative_symlink(src: Union[Path, str], dest: Union[Path, str]):
    subprocess.run(["ln", "-sr", src, dest])


def mkdir(path: Union[str, Path]):
    make_path(path).mkdir(parents=True, exist_ok=True)


def get_main_path() -> Path:
    return Path(Path(sys.argv[0])).resolve()
