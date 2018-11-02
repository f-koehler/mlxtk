import numpy


def get_operator_matrix(operator):
    nprod = operator.countProducts()
    grid_points = operator.pdim[0]

    matrix = numpy.zeros((grid_points, grid_points), dtype=numpy.complex128)
    unit_matrix = numpy.diag(numpy.ones(grid_points))

    for k in range(1, nprod + 1):
        # pylint: disable=protected-access
        term = operator._table.get((k, 1))
        coefficient = operator._table.get(k)

        if coefficient is None:
            coefficient = 1.0
        else:
            coefficient = coefficient.coef

        # unit operator
        if term is None:
            matrix += coefficient * unit_matrix
            continue

        # matrix operator
        if term.type == "matrix":
            if term.dim != grid_points:
                raise ValueError(
                    "invalid dimension {} of matrix term, expected {}".format(
                        term.dim, grid_points))
            matrix += coefficient * term.oterm.astype(numpy.complex128)
            continue

        # vector operator
        if term.type == "diag":
            if term.dim != grid_points:
                raise ValueError(
                    "invalid dimension {} of vector term, expected {}".format(
                        term.dim, grid_points))
            matrix += coefficient * numpy.diag(
                term.oterm.astype(numpy.complex128))
            continue

        # invalid type
        raise ValueError("invalid term type {}".format(term.type))

    return matrix
