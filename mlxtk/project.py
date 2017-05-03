import logging
import os

import mlxtk.operator
import mlxtk.wavefunction


class Project():
    """ML-MCTDH(X) project

    Attributes:
        operators (dict): dictionary containing operator creation functions
            index by names
        root_dir (str): root path for all files created during execution
    """

    def __init__(self, root_dir="."):
        self.operators = {}
        self.wavefunctions = {}
        self.root_dir = root_dir
        self.tasks = []

    def add_operator(self, name, func):
        """Add a new operator to the project

        Args:
            name (str): name of the operator
            func: parameterless function which returns the operator
        """
        self.operators[name] = func

    def add_wavefunction(self, name, func):
        self.wavefunctions[name] = func

    def add_task(self, task):
        self.tasks.append(task)

    def run(self):
        logging.basicConfig(level=logging.INFO)

        logging.info("Start project")
        if not os.path.exists(self.root_dir):
            logging.info("Create root path: %s)", self.root_dir)
            os.mkdir(self.root_dir)

        operator_path = os.path.join(self.root_dir, "operators")
        if not os.path.exists(operator_path):
            logging.info("Create operator path: %s", operator_path)
            os.mkdir(operator_path)
        operator_updated = {}
        for name in self.operators:
            logging.info("Generating operator: %s", name)
            op = self.operators[name]()

            path = os.path.join(operator_path, name + ".op")
            logging.info("Writing operator: %s -> %s", name, path)
            updated = mlxtk.operator.write_operator(op, path)

            operator_updated[name] = updated
            if updated:
                logging.info("Updated operator file: %s", path)
            else:
                logging.info("Operator \"%s\" is up-to-date, skip", name)

        wavefunction_path = os.path.join(self.root_dir, "wavefunctions")
        if not os.path.exists(wavefunction_path):
            logging.info("Create wavefunction path: %s", wavefunction_path)
            os.mkdir(wavefunction_path)
        wavefunction_updated = {}
        for name in self.wavefunctions:
            logging.info("Generating wave function: %s", name)
            wfn = self.wavefunctions[name]()

            path = os.path.join(wavefunction_path, name + ".wfn")
            logging.info("Writing wave function: %s -> %s", name, path)

            updated = mlxtk.wavefunction.write_wavefunction(wfn, path)
            wavefunction_updated[name] = updated
            if updated:
                logging.info("Updated wave function file: %s", path)
            else:
                logging.info("Wave function \"%s\" is up-to-date, skip", name)

        task_hash_path = os.path.join(self.root_dir, "task_hashes")
        if not os.path.exists(task_hash_path):
            logging.info("Create task hashes path: %s", task_hash_path)
            os.mkdir(task_hash_path)

        for task in self.tasks:
            task.root_dir = self.root_dir
            task.run()
            task.update_project(self)

    def __str__(self):
        return str(self.__dict__)
