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
    if name in DVRS:
        del DVRS[name]


def add_harmdvr(name, npoints, xeq, xho, tolerance=1e-15):
    if name in REGISTERED_DVRS:
        LOGGER.warn("overwrite existing dvr %s", name)
        remove_dvr(name)
    REGISTERED_DVRS[name] = [Harmdvr, npoints, xeq, xho, tolerance]


def add_rharmdvr(name, npoints, xeq, xho, tolerance=1e-15):
    if name in REGISTERED_DVRS:
        LOGGER.warn("overwrite existing dvr %s", name)
        remove_dvr(name)
    REGISTERED_DVRS[name] = [rHarmdvr, npoints, xeq, xho, tolerance]


def add_sindvr(name, npoints, qmin, qmax):
    if name in REGISTERED_DVRS:
        LOGGER.warn("overwrite existing dvr %s", name)
        remove_dvr(name)
    REGISTERED_DVRS[name] = [Sindvr, npoints, qmin, qmax]


def add_expdvr(name, npoints, qmin, qmax):
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


def get(name):
    if name not in DVRS:
        LOGGER.info("constructing dvr \"%s\" (%s)", name,
                    REGISTERED_DVRS[name][0].__name__)
        start = time.perf_counter()
        DVRS[name] = REGISTERED_DVRS[name][0](*REGISTERED_DVRS[name][1:])
        stop = time.perf_counter()
        LOGGER.info("execution took %fs", stop - start)

    return DVRS[name]
