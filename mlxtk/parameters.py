class Parameters(object):
    def __init__(self):
        self.parameter_names = []

    @staticmethod
    def init_from_dict(dictionary):
        parameters = Parameters()
        for key in dictionary:
            parameters.add_parameter(key, dictionary[key])
        return parameters

    def add_parameter(self, name, value):
        if name not in self.parameter_names:
            self.parameter_names.append(name)
        setattr(self, name, value)

    def set_all(self, *args):
        if len(args) != len(self.parameter_names):
            raise RuntimeError("Not all parameters specified")

        for i, name in enumerate(self.parameter_names):
            setattr(self, name, args[i])

    def to_list(self):
        result = []
        for name in self.parameter_names:
            result.append(getattr(self, name))
        return result

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def __str__(self):
        string = "Parameters {"
        max_length = len(max(self.parameter_names, key=lambda name: len(name)))
        for name in self.parameter_names:
            length = len(name)
            fill = ""
            if length < max_length:
                fill = " " * (max_length - length)
            string += "\n\t{}: {}{}".format(name, fill, getattr(self, name))
        return string + "\n}"
