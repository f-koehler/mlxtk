import pandas


def read_raw(path):
    # read the complete file
    data = pandas.read_csv(
        path,
        delimiter=r"\s+",
        header=None,
        names=["time", "x", "y", "real", "imaginary"],
    )

    # get a list of all time points
    times = data.time.unique().tolist()

    # create a data frame for each time point
    matrices = []
    for time in times:
        matrices.append(pandas.DataFrame(data.loc[data.time == time]))

    return matrices
