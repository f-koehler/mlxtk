import time

from QDTK.Primitive import Harmdvr
from QDTK.Primitive import rHarmdvr
from QDTK.Primitive import Sindvr
from QDTK.Primitive import Expdvr
from QDTK.Primitive import Legendredvr
from QDTK.Primitive import Laguerredvr
from QDTK.Primitive import Discrete
from QDTK.Primitive import FFT

from mlxtk import log

REGISTERED_DVRS = {}
DVRS = {}

LOGGER = log.get_logger(__name__)


def remove_dvr(name):
    """Remove a registered DVR

    Args:
        name (str): Name of the DVR
    """
    if name in DVRS:
        del DVRS[name]


def add_harmdvr(name, npoints, xeq, xho, tolerance=1e-15):
    """Register a new harmonic oscillator DVR

    Args:
        name (str): Name for the DVR object
        npoints (int): Number of grid points
        xeq (float): equilibrium position
        xh0 (float): harmonic oscillator length
        tolerance (float): ???
    """
    if name in REGISTERED_DVRS:
        LOGGER.warn("overwrite existing dvr %s", name)
        remove_dvr(name)
    REGISTERED_DVRS[name] = [Harmdvr, npoints, xeq, xho, tolerance]


def add_rharmdvr(name, npoints, xeq, xho, tolerance=1e-15):
    """Register a new radial harmonic oscillator DVR

    Args:
        name (str): Name for the DVR object
        npoints (int): Number of grid points
        xeq (float): equilibrium position
        xh0 (float): harmonic oscillator length
        tolerance (float): ???
    """
    if name in REGISTERED_DVRS:
        LOGGER.warn("overwrite existing dvr %s", name)
        remove_dvr(name)
    REGISTERED_DVRS[name] = [rHarmdvr, npoints, xeq, xho, tolerance]


def add_sinedvr(name, npoints, qmin, qmax):
    """Register a new sine DVR

    Args:
        name (str): Name for the DVR object
        npoints (int): Number of grid points
        qmin (float): Minimal x value
        qmax (float): Maximal x value
    """
    if name in REGISTERED_DVRS:
        LOGGER.warn("overwrite existing dvr %s", name)
        remove_dvr(name)
    REGISTERED_DVRS[name] = [Sindvr, npoints, qmin, qmax]


def add_expdvr(name, npoints, qmin, qmax):
    """Register a new exponential DVR

    Args:
        name (str): Name for the DVR object
        npoints (int): Number of grid points
        qmin (float): Minimal x value
        qmax (float): Maximal x value
    """
    if name in REGISTERED_DVRS:
        LOGGER.warn("overwrite existing dvr %s", name)
        remove_dvr(name)
    REGISTERED_DVRS[name] = [Expdvr, npoints, qmin, qmax]


def add_lengendredvr(name, npoints, m, tolerance=1e-10):
    if name in REGISTERED_DVRS:
        LOGGER.warn("overwrite existing dvr %s", name)
        remove_dvr(name)
    REGISTERED_DVRS[name] = [Legendredvr, npoints, m, tolerance]


def add_laguerredvr(name, npoints, alpha, xlag, x0, tolerance=1e-11):
    if name in REGISTERED_DVRS:
        LOGGER.warn("overwrite existing dvr %s", name)
        remove_dvr(name)
    REGISTERED_DVRS[name] = [Laguerredvr, npoints, alpha, xlag, x0, tolerance]


def add_discretedvr(name, nstates):
    if name in REGISTERED_DVRS:
        LOGGER.warn("overwrite existing dvr %s", name)
        remove_dvr(name)
    REGISTERED_DVRS[name] = [Discrete, nstates]


def add_fft(name, npoints, xmin, xmax):
    if name in REGISTERED_DVRS:
        LOGGER.warn("overwrite existing dvr %s", name)
        remove_dvr(name)
    REGISTERED_DVRS[name] = [FFT, npoints, xmin, xmax]


def exists(name):
    return name in REGISTERED_DVRS


def get(name):
    """Return the DVR registered under the given name

    Args:
        name (str): Name of the DVR

    Returns:
        The DVR object with the given name
    """
    if name not in DVRS:
        LOGGER.info("constructing dvr \"%s\" (%s)", name,
                    REGISTERED_DVRS[name][0].__name__)
        start = time.perf_counter()
        DVRS[name] = REGISTERED_DVRS[name][0](*REGISTERED_DVRS[name][1:])
        stop = time.perf_counter()
        LOGGER.info("execution took %fs", stop - start)

    return DVRS[name]


def is_fft(name):
    """Check wether a DVR is of FFT type

    Args:
        name (str): Name of the DVR

    Returns:
        bool: `True` if the DVR is of FFT type, `False` otherwise
    """
    if name not in REGISTERED_DVRS:
        raise RuntimeError("No DVR with name \"" + name + "\" present")

    return REGISTERED_DVRS[name][0] == FFT


def get_x(name):
    return get(name).x


def get_d1(name):
    if is_fft(name):
        return get(name).d1fft
    else:
        return get(name).d1dvr


def get_d2(name):
    if is_fft(name):
        return get(name).d2fft
    else:
        return get(name).d2dvr


def fftw_grid_sizes(n_max):
    sizes = []

    for i in range(2, n_max):
        n = i

        while True:
            if n % 5 == 0:
                n = n // 5
            else:
                break

        while True:
            if n % 3 == 0:
                n = n // 3
            else:
                break

        while True:
            if n % 2 == 0:
                n = n // 2
            else:
                break

        if n == 1:
            sizes.append(i)

    return sizes
