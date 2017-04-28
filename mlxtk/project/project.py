import mlxtk.hash


class Project:

    def __init__(self):
        self.parameters = {}
        self.grids = {}
        self.operators = {}

    # def is_outdated(self):
    #     for step in self.steps:
    #         if step.is_outdated():
    #             return True
    #     return False

    def hash_parameters(self):
        return mlxtk.hash.hash_dict(self.parameters)

    # def add_operator(name, operator_func):
    #     self.operators[name] = operator_func
