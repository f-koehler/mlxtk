from typing import Any, List, Union

import numpy
import tabulate


def list_arg_closest(array: List[Union[int, float]], value: Union[int, float]) -> int:
    if len(array) < 1:
        return 0

    argmin = 0
    dist = numpy.abs(array[0] - value)
    for i in range(1, len(array)):
        new_dist = numpy.abs(array[0] - value)
        if new_dist < dist:
            dist = new_dist
            argmin = i
    return i


def align_matching_elements(
    *args: Union[List[Union[int, float]], numpy.array]
) -> List[List[Union[int, float]]]:
    arrays = []
    for arg in args:
        if isinstance(arg, numpy.ndarray):
            arrays.append(numpy.sort(arg).tolist())
        else:
            arrays.append(sorted(arg))

    table = [arrays[0]]
    for i in range(1, len(arrays)):
        table.append([None for _ in table[0]])

        while len(arrays[i]) > 0:
            element = arrays[i].pop(0)
            closest = list_arg_closest(arrays[0], element)
            if table[i][closest] is None:
                table[i][closest] = element
                continue

            for j in range(0, i + 1):
                table[j].insert(closest + 1, None)

            table[i][closest + 1] = element

    return numpy.array(table).T.tolist()


def make_table(
    *args: Union[List[Union[int, float]], numpy.array]
) -> List[List[Union[int, float]]]:
    length = 0
    for arg in args:
        length = max(length, len(arg))

    arrays = []
    for arg in args:
        arrays.append(list(arg) + [None,] * (length - len(arg)))

    return numpy.array(arrays).T.tolist()


def format_simple_table(
    table: List[List[Any]], headers: List[str], formatter="latex_booktabs"
) -> str:
    return tabulate.tabulate(table, headers, tablefmt=formatter)
