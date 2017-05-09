from matplotlib import pyplot as plt
from mlxtk.task.task import SubTask
import mlxtk.inout.output as output
import mlxtk.inout.exp_val as exp_val
import mlxtk.log as log

import os.path


class EnergyPlot(SubTask):

    def __init__(self):
        SubTask.__init__(self, None)

        self.type = "EnergyPlot"
        self.name = "energy_plot"

        self.input_files = []
        self.output_files = ["energy.pdf"]
        self.symlinks = []

    def execute(self):
        log.info("Plot energy")

        data = output.read("output")
        plt.plot(data["time"], data["energy"])
        plt.savefig("energy.pdf")
        plt.close()


class NormPlot(SubTask):

    def __init__(self):
        SubTask.__init__(self, None)

        self.type = "NormPlot"
        self.name = "norm_plot"

        self.input_files = []
        self.output_files = ["norm.pdf"]
        self.symlinks = []

    def execute(self):
        log.info("Plot norm")

        data = output.read("output")
        plt.plot(data["time"], 1 - data["norm"])
        plt.savefig("norm.pdf")
        plt.close()


class OverlapPlot(SubTask):

    def __init__(self):
        SubTask.__init__(self, None)

        self.type = "OverlapPlot"
        self.name = "overlap_plot"

        self.input_files = []
        self.output_files = ["overlap.pdf"]
        self.symlinks = []

    def execute(self):
        log.info("Plot overlap")

        data = output.read("output")
        plt.plot(data["time"], data["overlap"])
        plt.savefig("overlap.pdf")
        plt.close()


class ExpectationValuePlot(SubTask):

    def __init__(self, observable):
        SubTask.__init__(self, None)

        self.type = "ExpectationValuePlot"
        self.name = "expectation_value_plot_" + observable
        self.observable = observable

        self.input_files = []
        self.output_files = [observable + ".pdf"]
        self.symlinks = []

    def execute(self):
        log.info("Plot expectation value of \"{}\"".format(self.observable))

        data = exp_val.read(self.observable + ".expval")

        f, ax = plt.subplots(2, 1, sharex=True)
        plt.sca(ax[0])
        plt.plot(data["time"], data["real"])
        plt.sca(ax[1])
        plt.plot(data["time"], data["imaginary"])
        plt.savefig(self.observable + ".pdf")
        plt.close()
