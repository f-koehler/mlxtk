def plot_entropy(ax, time, entropy):
    ax.plot(time, entropy)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$S_{\mathrm{B}}(t)$")
