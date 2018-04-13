import pandas


def read_basis_projection_ascii(path):
    with open(path) as fhandle:
        num_coeffs = (len(fhandle.readline().split()) - 1) // 3

    usecols = [0] + list(range(1, num_coeffs + 1)) + list(
        range(num_coeffs + 1, 2 * num_coeffs + 1))
    names = ["time"] + ["real_" + str(i) for i in range(num_coeffs)
                        ] + ["imag_" + str(i) for i in range(num_coeffs)]
    return pandas.read_csv(
        path, delim_whitespace=True, usecols=usecols, names=names)
