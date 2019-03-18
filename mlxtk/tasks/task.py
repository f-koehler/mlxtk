from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List


class Task(ABC):
    @abstractmethod
    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        pass

    def __call__(self) -> List[Callable[[], Dict[str, Any]]]:
        return self.get_tasks_run()
