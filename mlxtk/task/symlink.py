from . import task
from ..util import create_fresh_symlink


class SymlinkTask(task.Task):
    def __init__(self, src, dst, **kwargs):
        kwargs["task_type"] = "SymlinkTask"

        self.src = src
        self.dst = dst

        inp_src = task.ConstantInput("src", self.src)

        task.Task.__init__(
            self,
            "symlink_{}_{}".format(self.src, self.dst).replace("/", "_"),
            self.create_symlink,
            inputs=[inp_src],
            **kwargs)

    def create_symlink(self):
        create_fresh_symlink(self.src, self.dst)
