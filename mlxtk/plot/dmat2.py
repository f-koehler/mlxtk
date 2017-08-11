import mlxtk.plot.plot as plot

from matplotlib import pyplot
import numpy


def plot_dmat2(data_frame, dvr1=None, dvr2=None):
    p = plot.Plot()
    p.activate()

    pyplot.title("dmat2 (time = {})".format(data_frame.time[0]))
    n1 = len(data_frame.dof1.unique().tolist())
    n2 = len(data_frame.dof2.unique().tolist())

    if dvr1 and dvr2:
        weights1, weights2 = numpy.meshgrid(dvr1.weights, dvr2.weights)
        values = data_frame.element.values.reshape(n1, n2) * numpy.sqrt(
            weights1, weights2)
    else:
        values = data_frame.element.values.reshape(n1, n2)
    p.axes.pcolormesh(
        data_frame.dof1.values.reshape(n1, n2),
        data_frame.dof2.values.reshape(n1, n2), numpy.abs(values))

    p.axes.set_xlabel("dof 1")
    p.axes.set_ylabel("dof 2")

    return p
