#!/usr/bin/env python
from mlxtk.plot.plot_program import SimplePlotProgram, create_argparser
from mlxtk.inout.natpop import read_natpop


def main():
    parser = create_argparser(
        "Plot evolution of energy expectation value over time")
    parser.add_argument(
        "--in",
        type=str,
        dest="input_file",
        default="natpop",
        help="input_file (defaults to \"natpop\")")
    parser.add_argument(
        "--dof",
        type=int,
        default=1,
        help="degree of freedom for which to plot the natural populations")
    args = parser.parse_args()

    def init_plot(plot):
        data = read_natpop(args.input_file)[1][1]
        for column in data.columns[1:]:
            plot.axes.plot(data["time"], data[column]/1000.)
        plot.axes.set_xlabel("$t$")
        plot.axes.set_ylabel(r"$\lambda_i$")

    program = SimplePlotProgram("Energy", init_plot)
    program.main(args)


if __name__ == "__main__":
    main()
