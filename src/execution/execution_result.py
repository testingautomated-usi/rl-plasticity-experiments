from typing import Dict


class ExecutionResult:
    def __init__(
        self,
        adequate_performance: bool,
        info: Dict = None,
        regression: bool = True,
        regression_time: float = 0.0,
        training_time: float = 0.0,
        task_completed: bool = False,
    ):
        self.adequate_performance = adequate_performance
        # always true if adequate_performance = False
        self.regression = regression
        if not info:
            info = {}
        self.info = info
        self.task_completed = task_completed
        self.regression_time = regression_time
        self.training_time = training_time

    def get_info(self) -> Dict:
        return self.info

    def is_adequate_performance(self) -> bool:
        return self.adequate_performance

    def is_regression(self) -> bool:
        return self.regression

    def is_task_completed(self) -> bool:
        return self.task_completed

    def get_regression_time(self) -> float:
        return self.regression_time

    def get_training_time(self) -> float:
        return self.training_time
