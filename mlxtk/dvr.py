import numpy

from QDTK.Primitive import (
    FFT,
    Expdvr,
    Harmdvr,
    Laguerredvr,
    Legendredvr,
    Sindvr,
    rHarmdvr,
)

# from QDTK.Primitive import Discrete
# from QDTK.Primitive import FFT

DVR_CLASSES = {
    "HarmonicDVR": Harmdvr,
    "RadialHarmonicDVR": rHarmdvr,
    "SineDVR": Sindvr,
    "ExponentialDVR": Expdvr,
    "LegendreDVR": Legendredvr,
    "LaguerreDVR": Laguerredvr,
    "FFT": FFT,
}

for dvr_class in DVR_CLASSES:

    def dummy_d4(self):
        del self

    setattr(DVR_CLASSES[dvr_class], "_calcd4dvr", dummy_d4)


class DVRSpecification:
    """A specification for a DVR grid

    Args:
       type_ (str): type of the DVR
       *args (list): list of arguments to construct the DVR

    Attributes:
       type_ (str): type of the DVR
       args (list): list of arguments to construct the DVR
       dvr: the DVR object once it is constructed
    """

    def __init__(self, type_: str, *args):
        self.type_ = type_
        self.args = args
        self.dvr = None  # type: Dvr

    def compute(self):
        if self.dvr is None:
            class_ = DVR_CLASSES[self.type_]
            self.dvr = class_(*self.args)

    def get(self):
        self.compute()
        return self.dvr

    def get_x(self) -> numpy.ndarray:
        self.compute()
        return self.dvr.x

    def get_weights(self) -> numpy.ndarray:
        self.compute()
        return self.dvr.weights

    def get_d1(self) -> numpy.ndarray:
        self.compute()
        return self.dvr.d1dvr

    def get_d2(self) -> numpy.ndarray:
        self.compute()
        return self.dvr.d2dvr

    def get_delta(self) -> numpy.ndarray:
        self.compute()
        return self.dvr.delta_w()

    def is_fft(self) -> bool:
        return self.type_ == "FFT"

    def __getstate__(self):
        return {"type": self.type_, "args": self.args}

    def __setstate__(self, state):
        if not hasattr(self, "type_"):
            self.dvr = None
        elif not hasattr(self, "args_"):
            self.dvr = None
        elif (state["type"] != self.type_) or (state["args"] != self.args):
            self.dvr = None

        self.type_ = state["type"]
        self.args = state["args"]


def add_harmdvr(
    npoints: int, xeq: float, xho: float, tolerance: float = 1e-15
) -> DVRSpecification:
    """Register a new harmonic oscillator DVR

    Args:
        npoints (int): lnumber of grid points
        xeq (float): equilibrium position
        xh0 (float): harmonic oscillator length
        tolerance (float): ???
    """
    return DVRSpecification("HarmonicDVR", npoints, xeq, xho, tolerance)


def add_rharmdvr(
    npoints: int, xeq: float, xho: float, tolerance: float = 1e-15
) -> DVRSpecification:
    """Register a new radial harmonic oscillator DVR

    Args:
        npoints (int): number of grid points
        xeq (float): equilibrium position
        xh0 (float): harmonic oscillator length
        tolerance (float): ???
    """
    return DVRSpecification("RadialHarmonicDVR", npoints, xeq, xho, tolerance)


def add_sinedvr(npoints: int, qmin: float, qmax: float) -> DVRSpecification:
    """Register a new sine DVR

    Args:
        npoints (int): number of grid points
        qmin (float): minimal x value
        qmax (float): maximal x value
    """
    return DVRSpecification("SineDVR", npoints, qmin, qmax)


def add_expdvr(npoints: int, qmin: float, qmax: float) -> DVRSpecification:
    """Register a new exponential DVR

    Args:
        npoints (int): number of grid points
        qmin (float): minimal x value
        qmax (float): maximal x value
    """
    return DVRSpecification("ExponentialDVR", npoints, qmin, qmax)


def add_lengendredvr(
    npoints: int, m: int, tolerance: float = 1e-10
) -> DVRSpecification:
    return DVRSpecification("LegendreDVR", npoints, m, tolerance)


def add_laguerredvr(
    npoints: int, alpha: float, xlag: float, x0: float, tolerance: float = 1e-11
) -> DVRSpecification:
    return DVRSpecification("LaguerreDVR", npoints, alpha, xlag, x0, tolerance)


def add_fft(npoints: int, xmin: float, xmax: float) -> DVRSpecification:
    return DVRSpecification("FFT", npoints, xmin, xmax)


# def add_discretedvr(name, nstates):
#     if name in REGISTERED_DVRS:
#         LOGGER.warn("overwrite existing dvr %s", name)
#         remove_dvr(name)
#     REGISTERED_DVRS[name] = [Discrete, nstates]

# def add_fft(name, npoints, xmin, xmax):
#     if name in REGISTERED_DVRS:
#         LOGGER.warn("overwrite existing dvr %s", name)
#         remove_dvr(name)
#     REGISTERED_DVRS[name] = [FFT, npoints, xmin, xmax]

# def exists(name):
#     return name in REGISTERED_DVRS

# def get(name):
#     """Return the DVR registered under the given name

#     Args:
#         name (str): Name of the DVR

#     Returns:
#         The DVR object with the given name
#     """
#     if name not in DVRS:
#         LOGGER.info("constructing dvr \"%s\" (%s)", name,
#                     REGISTERED_DVRS[name][0].__name__)
#         start = time.perf_counter()
#         DVRS[name] = REGISTERED_DVRS[name][0](*REGISTERED_DVRS[name][1:])
#         stop = time.perf_counter()
#         LOGGER.info("execution took %fs", stop - start)

#     return DVRS[name]

# def is_fft(name):
#     """Check wether a DVR is of FFT type

#     Args:
#         name (str): Name of the DVR

#     Returns:
#         bool: `True` if the DVR is of FFT type, `False` otherwise
#     """
#     if name not in REGISTERED_DVRS:
#         raise RuntimeError("No DVR with name \"" + name + "\" present")

#     return REGISTERED_DVRS[name][0] == FFT

# def get_x(name):
#     return get(name).x

# def get_d1(name):
#     if is_fft(name):
#         return get(name).d1fft
#     else:
#         return get(name).d1dvr

# def get_d2(name):
#     if is_fft(name):
#         return get(name).d2fft
#     else:
#         return get(name).d2dvr

# def fftw_grid_sizes(n_max):
#     sizes = []

#     for i in range(2, n_max):
#         n = i

#         while True:
#             if n % 5 == 0:
#                 n = n // 5
#             else:
#                 break

#         while True:
#             if n % 3 == 0:
#                 n = n // 3
#             else:
#                 break

#         while True:
#             if n % 2 == 0:
#                 n = n // 2
#             else:
#                 break

#         if n == 1:
#             sizes.append(i)

#     return sizes
