import os.path
from pathlib import Path
from typing import List

import numpy


def labels_from_paths(paths: List[str]) -> List[str]:
    common = os.path.commonpath(paths)
    parts = [Path(os.path.relpath(p, common)).parts for p in paths]
    arr = numpy.full(
        (len(parts), len(max(parts, key=lambda p: len(p)))), "", dtype=object)
    for i, _ in enumerate(parts):
        for j, _ in enumerate(parts[i]):
            arr[i][j] = parts[i][j]
    mask = [
        i for i in range(arr.shape[1]) if numpy.unique(arr[:, i]).shape[0] != 1
    ]
    arr = arr[:, mask]
    return ["/".join(a) for a in arr]
