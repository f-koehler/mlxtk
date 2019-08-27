from typing import List, Union

import numpy


def list_arg_closest(array: List[Union[int, float]], value: Union[int, float]):
    return numpy.abs(numpy.array(array) - value).argmin()


def align_matching_elements(*args: Union[List[Union[int, float]], numpy.array]
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

    return table
