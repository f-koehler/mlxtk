def add_argparse_2d_args(parser):
    parser.add_argument(
        "--logx",
        action="store_true",
        dest="logx",
        help="use log scale on the x-axis")
    parser.add_argument(
        "--no-logx",
        action="store_false",
        dest="logx",
        help="do not use log scale on the x-axis", )
    parser.add_argument(
        "--logy",
        action="store_true",
        dest="logy",
        help="use log scale on the y-axis")
    parser.add_argument(
        "--no-logy",
        action="store_false",
        dest="logy",
        help="do not use log scale on the y-axis", )
    parser.add_argument("--xmin", type=float, help="minimum for the x axis")
    parser.add_argument("--xmax", type=float, help="maximum for the x axis")
    parser.add_argument("--ymin", type=float, help="minimum for the y axis")
    parser.add_argument("--ymax", type=float, help="maximum for the y axis")
    parser.add_argument(
        "--grid", action="store_true", dest="grid", help="draw a grid")
    parser.add_argument(
        "--no-grid",
        action="store_false",
        dest="grid",
        help="do not draw a grid")


def apply_2d_args(ax, args):
    if args.logx:
        ax.set_xscale("log")

    if args.logy:
        ax.set_yscale("log")

    xlims = ax.get_xlim()
    if args.xmin is not None:
        xlims[0] = args.xmin
    if args.xmax is not None:
        xlims[1] = args.xmax
    ax.set_xlim(xlims)

    ylims = ax.get_ylim()
    if args.ymin is not None:
        ylims[0] = args.ymin
    if args.ymax is not None:
        ylims[1] = args.ymax
    ax.set_ylim(ylims)

    ax.grid(args.grid)
