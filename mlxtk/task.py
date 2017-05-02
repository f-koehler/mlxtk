import os.path


class Relaxation(Task):

    def __init__(self, initial_wavefunction, final_wavefunction, root_dir):
        self.initial_wavefunction = initial_wavefunction
        self.final_wavefunction = final_wavefunction

    def is_up_to_date(self):
        input_path = self.get_input_path()
        output_path = self.get_output_path()

        if not os.path.exists(input_path):
            raise RuntimeError("Wavefunction \"\" does not exist (\"\")".format(
                initial_wavefunction, input_path))

        if not os.path.exists(output_path):
            return False

        return os.path.getmtime(output_path) > os.path.getmtime(input_path)

    def get_input_path(self):
        return os.path.join(root_dir, "wavefunctions",
                            initial_wavefunction + ".wfn")

    def get_output_path(self):
        return os.path.join(root_dir, "wavefunctions",
                            final_wavefunction + ".wfn")
