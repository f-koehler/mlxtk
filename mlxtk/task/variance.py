import h5py
import numpy
import os
import pandas

from . import task
from mlxtk.inout import hdf5
from mlxtk.inout.expval import read_expval, add_variance_to_hdf5


class ComputeVarianceTask(task.Task):
    def __init__(self, propagation, operator, operator_squared, **kwargs):
        """Task to calculate the time-dependent variance of an operator

        Calculates the variance :math:`\langle{(O-\langle O\rangle)}^2\rangle=\langle O^2\rangle-{\langle O\rangle}^2`
        based on the expectation values :math:\langle O\rangle` and `\langle O^2 \rangle

        Args:
            propagation (mlxtk.task.PropagationTask): Propagation that creates the psi file
            operator (str): name of the operator O
            operator_squared (str): name of the squared operator O^2

        Attributes:
            propagation (mlxtk.task.PropagationTask): Propagation that creates the psi file
            operator (str): name of the operator O
            operator_squared (str): name of the squared operator O^2
            path_expval_operator (str): path to the expectation value file <O>
            path_expval_operator_squared (str): path to the expectation value file <O^2>
        """
        kwargs["task_type"] = "ComputeVarianceTask"

        self.propagation = propagation
        self.operator = operator
        self.operator_squared = operator_squared

        self.path_expval_operator = os.path.join(propagation.propagation_name,
                                                 operator + ".expval")
        self.path_expval_operator_squared = os.path.join(
            propagation.propagation_name, operator_squared + ".expval")
        self.path_variance = os.path.join(propagation.propagation_name,
                                          operator + ".variance")

        inp_expval_operator = task.FileInput("expval_operator",
                                             self.path_expval_operator)
        inp_expval_operator_squared = task.FileInput(
            "expval_operator_squared", self.path_expval_operator_squared)

        out_variance = task.FileOutput("variance", self.path_variance)

        task.Task.__init__(
            self,
            propagation.propagation_name + "_variance_" + operator,
            self.compute_variance,
            inputs=[inp_expval_operator, inp_expval_operator_squared],
            outputs=[out_variance],
            **kwargs)

    def compute_variance(self):
        data_operator = read_expval(self.path_expval_operator)
        data_operator_squared = read_expval(self.path_expval_operator_squared)

        variance = (data_operator_squared.real +
                    1j * data_operator_squared.imaginary) - (
                        data_operator.real + 1j * data_operator.imaginary)**2

        data_variance = pandas.DataFrame(
            numpy.column_stack((data_operator.time, numpy.real(variance),
                                numpy.imag(variance))),
            columns=["time", "real", "imaginary"],
        )

        data_variance.to_csv(
            self.path_variance, sep="\t", index=False, header=False)

    def create_hdf5(self, group=None):
        if self.is_running():
            raise hdf5.IncompleteHDF5(
                "Possibly running ComputeVarianceTask, no data can be added")

        opened_file = group is None
        if opened_file:
            self.logger.info("create new hdf5 file")
            group = h5py.File(self.propagation.propagation_name + ".hdf5", "w")

        # TODO: this fails, but something like it should be there
        # else:
        #     if self.propagation.propagation_name in group:
        #         group = group[self.propagation.propagation_name]
        #     else:
        #         group = group.create_group(self.propagation.propagation_name)

        if not os.path.exists(self.path_variance):
            raise FileNotFoundError(
                "Variance file \"" + self.path_variance + "\" does not exist")

        add_variance_to_hdf5(group, self.path_variance)

        if opened_file:
            group.close()
