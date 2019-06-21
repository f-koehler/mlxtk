from typing import Any, Dict, Tuple

import numpy

from QDTK.Primitive import (FFT, Dvr, Expdvr, Harmdvr, Laguerredvr,
                            Legendredvr, Sindvr, rHarmdvr)

assert Any
assert Dict
assert Tuple

DVR_CLASSES = {
    "HarmonicDVR": Harmdvr,
    "RadialHarmonicDVR": rHarmdvr,
    "SineDVR": Sindvr,
    "ExponentialDVR": Expdvr,
    "LegendreDVR": Legendredvr,
    "LaguerreDVR": Laguerredvr,
    "FFT": FFT,
}
DVR_CACHE = {CLASS: {}
             for CLASS in DVR_CLASSES
             }  # type: Dict[Any, Dict[Tuple[Any], Dvr]]

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

        if (type_ == "ExponentialDVR") and (args[0] % 2 == 0):
            raise RuntimeError(
                "ExponentialDVR requires an odd number of grid points")

    def compute(self):
        """Compute the dvr.

        This creates the corresponding :py:class:`QDTK.Primitive.Dvr` object
        if it does not exist in the cache variable
        :py:data:`mlxtk.dvr.DVR_CACHE`
        """
        if self.dvr is not None:
            return

        if self.args in DVR_CACHE[self.type_]:
            self.dvr = DVR_CACHE[self.type_][self.args]
        else:
            class_ = DVR_CLASSES[self.type_]
            self.dvr = class_(*self.args)
            DVR_CACHE[self.type_][self.args] = self.dvr

    def get(self) -> Dvr:
        """Return the associated :py:class:`QDTK.Primitive.Dvr` object.

        Returns:
            The DVR object.
        """
        self.compute()
        return self.dvr

    def get_x(self) -> numpy.ndarray:
        """Return the x operator with respect to the primitive basis.

        Returns:
            The x operator.
        """
        self.compute()
        return self.dvr.x

    def get_weights(self) -> numpy.ndarray:
        """Return the DVR weights.

        Returns:
            The DVR weights.
        """
        self.compute()
        return self.dvr.weights

    def get_d1(self) -> numpy.ndarray:
        """Return the first derivative with respect to the primitive basis.

        Returns:
            The first spatial derivative.
        """
        self.compute()
        if self.is_fft():
            return self.dvr.d1fft
        return self.dvr.d1dvr

    def get_d2(self) -> numpy.ndarray:
        """Return the second derivative with respect to the primitive basis.

        Returns:
            The second spatial derivative.
        """
        self.compute()
        if self.is_fft():
            return self.dvr.d2fft
        return self.dvr.d2dvr

    def get_delta(self) -> numpy.ndarray:
        """Return the Dirac delta with respect to the primitive basis.

        Returns:
            The dirac delta.
        """
        self.compute()
        return self.dvr.delta_w()

    def is_fft(self) -> bool:
        """Check whether this DVR is a FFT.

        Returns:
            :code:`True` if the DVR is a FFT, :code:`False` otherwise.
        """
        return self.type_ == "FFT"

    def get_expdvr(self):
        """Add an exponential DVR.

        This is a workaround as :py:mod:`QDTK` does not allow the
        diagonalization of one-body operators that use FFT.

        Returns:
            The :py:class:`mxltk.dvr.DVRSpecification` for the corresponding
            exponential DVR.
        """
        if not self.is_fft():
            raise RuntimeError("get_expdvr() makes only sense for a FFT grid")

        return add_expdvr(self.args[0], self.args[1], self.args[2])

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

    def __str__(self) -> str:
        return ("DVRSpecification<" + self.type_ + ", " + ", ".join(
            (str(arg) for arg in self.args)) + ">")


def add_harmdvr(npoints: int, xeq: float, xho: float,
                tolerance: float = 1e-15) -> DVRSpecification:
    """Register a new harmonic oscillator DVR

    Args:
        npoints (int): lnumber of grid points
        xeq (float): equilibrium position
        xh0 (float): harmonic oscillator length
        tolerance (float): ???
    """
    return DVRSpecification("HarmonicDVR", npoints, xeq, xho, tolerance)


def add_rharmdvr(npoints: int,
                 xeq: float,
                 xho: float,
                 tolerance: float = 1e-15) -> DVRSpecification:
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


def add_lengendredvr(npoints: int, m: int,
                     tolerance: float = 1e-10) -> DVRSpecification:
    return DVRSpecification("LegendreDVR", npoints, m, tolerance)


def add_laguerredvr(npoints: int,
                    alpha: float,
                    xlag: float,
                    x0: float,
                    tolerance: float = 1e-11) -> DVRSpecification:
    return DVRSpecification("LaguerreDVR", npoints, alpha, xlag, x0, tolerance)


def add_fft(npoints: int, xmin: float, xmax: float) -> DVRSpecification:
    return DVRSpecification("FFT", npoints, xmin, xmax)
