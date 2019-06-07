def plot_natpop(ax, time, natpop):
    for i in range(natpop.shape[1]):
        ax.plot(time, natpop[:, i])
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\lambda_i(t)$")


def plot_depletion(ax, time, natpop):
    ax.plot(time, 1 - natpop[:, 0])
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$1-\lambda_1(t)$")
