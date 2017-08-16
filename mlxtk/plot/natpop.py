from mlxtk.plot.plot import Plot


def plot_natpop(data):
    node = 1
    orbitals = 0

    num_orbitals = len(data[node][orbitals].columns) - 1

    plt = Plot()

    for i in range(0, num_orbitals):
        plt.axes.plot(data[node][orbitals].time,
                      data[node][orbitals]["orbital" + str(i)])

    return plt
