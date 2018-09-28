from doit.cmd_base import TaskLoader
from doit.task import dict_to_task
from doit.doit_cmd import DoitMain


class CustomTaskLoader(TaskLoader):
    def __init__(self, tasks):
        self.tasks = tasks

    def load_tasks(self, *args):
        tasks = []
        for task_dict in self.tasks:
            tasks.append(dict_to_task(task_dict))
        return tasks, {}


def run_doit(task_generators, arguments=[]):
    return DoitMain(CustomTaskLoader(task_generators)).run(arguments)
