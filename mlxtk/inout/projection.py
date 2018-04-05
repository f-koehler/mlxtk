import pandas

def read_td_basis_projection_ascii(path):
    with open(path) as fhandle:
        print(fhandle.readline())
        exit()
        num_coeffs = (len(fhandle.readline().split()) - 1) // 2
    print(num_coeffs)
    # pandas.read_csv(path, delim_whitespace=True,
