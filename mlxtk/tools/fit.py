from scipy.optimize import curve_fit
import numpy


class CosineFit(object):
    """Fit a cosine function to a set of points

    A function of type :math:`f(x)=a\cdot\cos(b*x+c)+d` is fitted to the given
    points using :func:`scipy.optimize.curve_fit`.
    This object tries to extract good initial values for the fit parameters
    from the data.
    This also yields an estimate for the jacobian of the fit functions which
    is then used to accelerate the optimization routine.
    Furthermore, the ranges for the parameters are used to bound the search
    space: :math:`a\in\mathbb{R}`, :math:`b\in\left[0,\infty\right)`,
    :math:`c\in\left[0,w\pi\right)` and :math:`d\in\mathbb{R}`
    """
    def __init__(self, x, y, **curve_fit_args):
        self.y_offset = numpy.mean(y)
        self.y_amplitude = (numpy.max(y) - numpy.min(y)) / 2.

        y_rel = y - self.y_offset
        sign = numpy.sign(y_rel)
        sign_zero = (sign == 0)
        while sign_zero.any():
            sign[sign_zero] = numpy.roll(sign_zero, 1)[sign_zero]
            sign_zero = (sign == 0)
        self.y_sign_changes = numpy.nonzero(
            ((numpy.roll(sign, 1) - sign) != 0))[0]
        self.y_roots = x[self.y_sign_changes]

        self.y_period = 2 * (self.y_roots[-1] - self.y_roots[0]) / (
            len(self.y_roots) - 1)
        self.y_angular_frequency = 2 * numpy.pi / self.y_period

        self.y_phase = numpy.pi / 2. - (
            self.y_roots[0] * self.y_angular_frequency)
        two_pi = 2 * numpy.pi
        while self.y_phase < 0.:
            self.y_phase += two_pi
        while self.y_phase >= two_pi:
            self.y_phase -= two_pi

        self.popt, self.pcov = curve_fit(
            CosineFit.fit_function,
            x,
            y,
            p0=(self.y_amplitude, self.y_angular_frequency, self.y_phase,
                self.y_offset),
            bounds=((-numpy.inf, 0., 0., -numpy.inf),
                    (numpy.inf, numpy.inf, 2 * numpy.pi, numpy.inf)),
            jac=CosineFit.fit_jacobian,
            **curve_fit_args)

    def __call__(self, x):
        return CosineFit.fit_function(x, self.popt[0], self.popt[1],
                                      self.popt[2], self.popt[3])

    def jacobian(self, x):
        return CosineFit.fit_jacobian(x, self.popt[0], self.popt[1],
                                      self.popt[2], self.popt[3])

    def guess(self, x):
        return CosineFit.fit_function(x, self.y_amplitude,
                                      self.y_angular_frequency, self.y_phase,
                                      self.y_offset)

    def guess_jacobian(self, x):
        return CosineFit.fit_jacobian(x, self.y_amplitude,
                                      self.y_angular_frequency, self.y_phase,
                                      self.y_offset)

    def parameters(self):
        return self.popt

    def errors(self):
        return numpy.sqrt(numpy.diag(self.pcov))

    @staticmethod
    def fit_function(t, amplitude, angular_frequency, phase, offset):
        return amplitude * numpy.cos(angular_frequency * t + phase) + offset

    @staticmethod
    def fit_jacobian(t, amplitude, angular_frequency, phase, offset):
        return numpy.column_stack(
            (numpy.cos(angular_frequency * t + phase),
             -amplitude * t * numpy.sin(angular_frequency * t + phase),
             -amplitude * numpy.sin(angular_frequency * t + phase),
             numpy.full(t.shape, 1.)))
