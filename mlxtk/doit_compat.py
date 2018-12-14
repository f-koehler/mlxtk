from doit.cmd_base import TaskLoader
from doit.doit_cmd import DoitMain
from doit.task import dict_to_task

from .timing import Timer


class CustomTaskLoader(TaskLoader):
    def __init__(self, task_generators):
        self.task_generators = task_generators

    def load_tasks(self, cmd, opt_values, pos_args):
        del cmd
        del opt_values
        del pos_args
        tasks = []
        for task_generator in self.task_generators:
            tasks.append(dict_to_task(task_generator()))
        return tasks, {}


def run_doit(task_generators, arguments=[]):
    return DoitMain(CustomTaskLoader(task_generators)).run(arguments)


class DoitAction:
    def __init__(self, func):
        self.func = func

    def __call__(self, targets, *args, **kwargs):
        timer = Timer()
        ret = self.func(targets, *args, **kwargs)
        timer.stop()

        if ret is None:
            return {
                self.func.__name__: {
                    "monotonic_time": timer.get_monotonic_time(),
                    "perf_time": timer.get_perf_time(),
                    "process_time": timer.get_process_time(),
                }
            }

        if isinstance(ret, dict):
            ret[self.func.__name__] = ret.get(self.func.__name__, {})
            ret[self.func.__name__][
                "monotonic_time"] = timer.get_monotonic_time()
            ret[self.func.__name__]["perf_time"] = timer.get_perf_time()
            ret[self.func.__name__]["process_time"] = timer.get_process_time()
            return ret

        if isinstance(ret, bool):
            if not ret:
                return ret
            return {
                self.func.__name__: {
                    "monotonic_time": timer.get_monotonic_time(),
                    "perf_time": timer.get_perf_time(),
                    "process_time": timer.get_process_time(),
                }
            }

        raise NotImplementedError(
            "The return type {} is not supported for Doit actions")
