def get_defaults():
    return {"logx": False, "logy": False, "grid": True, "figsize": "4.5,4.5"}


def parse_figsize(size):
    return tuple(float(x) for x in size.split(","))


def add_plotting_arguments(parser, defaults=get_defaults()):
    parser.add_argument(
        "--out",
        type=str,
        dest="output_file",
        help="output file for the plot, also disables the GUI",
    )
    parser.add_argument(
        "--logx",
        action="store_true",
        default=defaults["logx"],
        help="logarithmic x axis",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        default=defaults["logy"],
        help="logarithmic y axis",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        default=not defaults["grid"],
        help="disable grid",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        help=
        "minimal value on the x-axis, automatically chosen if not specified",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        help=
        "maximal value on the x-axis, automatically chosen if not specified",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        help=
        "minimal value on the y-axis, automatically chosen if not specified",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        help=
        "maximal value on the y-axis, automatically chosen if not specified",
    )
    parser.add_argument("--figsize", type=str, default=defaults["figsize"])
    return parser
