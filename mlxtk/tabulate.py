from tabulate import tabulate


def create_tabulate_argument(parser, argument_names, destination):
    parser.add_argument(
        *argument_names,
        dest=destination,
        type=str,
        default="plain",
        choices=[
            "plain", "simple", "grid", "fancy_grid", "pip", "orgtbl", "jira",
            "presto", "psql", "rst", "mediawiki", "moinmoin", "youtrack",
            "html", "latex", "latex_raw", "latex_booktabs", "textile"
        ], metavar=destination,
        help="format of the parameter table (see https://bitbucket.org/astanin/python-tabulate for choices)")
