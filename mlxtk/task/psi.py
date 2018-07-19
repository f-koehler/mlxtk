from .task import Task, FileInput, ConstantInput, FileOutput

import numpy


class PsiTimeSlice(Task):
    def __init__(self, psi_in, psi_out, t_min, t_max, step=1, **kwargs):
        kwargs["task_type"] = "PsiTimeSlice"

        self.psi_in = psi_in
        self.psi_out = psi_out
        self.t_min = t_min
        self.t_max = t_max
        self.step = step

        inp_psi = FileInput("psi_in", self.psi_in)
        inp_t_min = ConstantInput("t_min", t_min)
        inp_t_max = ConstantInput("t_max", t_max)
        inp_step = ConstantInput("step", step)
        out_psi = FileOutput("psi_out", self.psi_out)

        name = "psi_time_slice_{}_{}".format(self.psi_in,
                                             self.psi_out).replace("/", "_")

        Task.__init__(
            self,
            name,
            self.create_time_slice,
            inputs=[inp_psi, inp_t_min, inp_t_max, inp_step],
            outputs=[out_psi],
            **kwargs)

    def create_time_slice(self):
        with open(self.psi_in, "r") as fhin:
            with open(self.psi_out, "w") as fhout:
                # read tape
                line = fhin.readline()
                if not line.startswith("$tape"):
                    raise RuntimeError(
                        "expected \"$tape\" tag at beginning of file")

                while True:
                    if line == "":
                        raise RuntimeError(
                            "unexpected end of file while reading tape")
                    if line.strip() == "":
                        break
                    fhout.write(line)
                    line = fhin.readline()

                current_step = 0

                # read actual data
                while True:
                    line = fhin.readline()
                    if line == "":
                        return

                    if not line.startswith("$time"):
                        raise RuntimeError("expected \"$time\" tag")
                    line = fhin.readline()
                    time = float(line.strip().split(" ")[0])
                    if (time < self.t_min) or (time > self.t_max):
                        skipping = True
                    else:
                        if current_step == 0:
                            skipping = False
                            fhout.write("\n$time\n" + line)
                        else:
                            skipping = True
                        current_step = (current_step + 1) % self.step

                    while True:
                        line = fhin.readline()

                        # check if we have reached the end of file
                        if line == "":
                            return

                        # check if we have reached the empty line before
                        # the next time
                        if line.strip() == "":
                            break

                        if not skipping:
                            fhout.write(line)


class ExtractWaveFunction(Task):
    @staticmethod
    def compose_name(psi, time):
        return "extract_wfn_{}_{}".format(psi, time).replace("/", "_")

    def __init__(self, wfn_out, psi, time, **kwargs):
        kwargs["task_type"] = "ExtractWaveFunction"

        self.wfn_out = wfn_out
        self.psi = psi
        self.time = time

        inp_psi = FileInput("psi", self.psi)
        inp_time = ConstantInput("time", self.time)
        out_wfn = FileOutput("wfn_out", wfn_out)

        Task.__init__(
            self,
            ExtractWaveFunction.compose_name(psi, time),
            self.extract_wave_function,
            inputs=[inp_psi, inp_time],
            outputs=[out_wfn],
            **kwargs)

    def extract_wave_function(self):
        with open(self.psi, "r") as fhin:
            with open(self.wfn_out + ".wfn", "w") as fhout:
                # read tape
                line = fhin.readline()
                if not line.startswith("$tape"):
                    raise RuntimeError(
                        "expected \"$tape\" tag at beginning of file")

                while True:
                    if line == "":
                        raise RuntimeError(
                            "unexpected end of file while reading tape")
                    if line.strip() == "":
                        break
                    fhout.write(line)
                    line = fhin.readline()

                fhout.write(line)

                # read actual data
                while True:
                    line = fhin.readline()
                    if line == "":
                        return

                    if not line.startswith("$time"):
                        raise RuntimeError("expected \"$time\" tag")
                    line = fhin.readline()
                    time = float(line.strip().split(" ")[0])

                    if numpy.abs(time - self.time) < 1e-10:
                        fhout.write("$time\n")
                        fhout.write(line)

                        while True:
                            line = fhin.readline()

                            # check if we have reached the end of file
                            if line == "":
                                return

                            # check if we reached the next time
                            if line.strip() == "":
                                return

                            fhout.write(line)
                    else:
                        while True:
                            line = fhin.readline()

                            # check if we have reached the end of file
                            if line == "":
                                return

                            # check if we reached the next time
                            if line.strip() == "":
                                break
