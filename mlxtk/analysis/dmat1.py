import numpy
import pandas


def compute_trace(data, dvr=None):
    if dvr:
        weights_x, weights_y = numpy.meshgrid(
            numpy.sqrt(dvr.weights), numpy.sqrt(dvr.weights))

    def analyze_frame(frame):
        n = len(frame.x.unique().tolist())
        data = (frame.real.values + frame.imaginary.values).reshape(n, n)
        if dvr:
            data *= weights_x * weights_y
        return numpy.trace(data)

    if isinstance(data, pandas.DataFrame):
        return analyze_frame(data)
    else:
        results = []
        for frame in data:
            results.append(analyze_frame(frame))
        return numpy.array(results)
