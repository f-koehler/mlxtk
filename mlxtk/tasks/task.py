from abc import ABC, abstractmethod

class Task(ABC):
    @abstractmethod
    def get_tasks_run():
        pass

    def __call__(self):
        return self.get_tasks_run()
