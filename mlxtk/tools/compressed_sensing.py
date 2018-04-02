import numpy
import cvxpy


def compsens1d(t, f, w_max, dw):
    dt = t[1] - t[0]
    eta = dt * dw
    N_t = len(t)
    N_w = int(w_max / dw + 1.)
    assert N_t < 2 * N_w
    products = numpy.outer(numpy.arange(0, N_t), numpy.arange(0, N_w))
    f_norm = numpy.sum(numpy.abs(f))
    f_tilde = f / f_norm
    A = (2. * dw / numpy.pi) * numpy.block(
        [numpy.cos(eta * products),
         numpy.sin(eta * products)])
    variable = cvxpy.Variable(2 * N_w)
    objective = cvxpy.Minimize(cvxpy.norm(variable, 1))
    constraint = [cvxpy.norm(A * variable - f_tilde, 2) <= 1e-10]
    problem = cvxpy.Problem(objective, constraint)
    problem.solve(verbose=True)
    coeffs = (
        numpy.sqrt(numpy.pi / 2.) * f_norm / dw) * numpy.array(variable.value)
    freq = numpy.arange(0., w_max + dw, dw)
    coeffs = coeffs[0:N_w] + 1j * coeffs[N_w:], (N_w)
    return freq, coeffs
