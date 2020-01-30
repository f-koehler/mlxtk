#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy
from matplotlib.widgets import Slider

tensor = numpy.load("delta_spf.npy")
m = tensor.shape[0]

fig, ax = plt.subplots(1, 1)
tensor = numpy.abs(tensor)
Y, X = numpy.meshgrid(numpy.arange(m), numpy.arange(m))
ax.imshow(numpy.reshape(tensor, (m * m, m * m)))
plt.show()
