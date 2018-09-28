def plot_natpop(ax, time, natpop):
    for i in range(natpop.shape[1]):
        ax.plot(time, natpop[:, i])
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\lambda_i(t)$")
