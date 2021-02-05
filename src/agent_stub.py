import os
import random
import time
from queue import Queue
from typing import Tuple

from abstract_agent import AbstractAgent
from algo.env_predicate_pair import EnvPredicatePair
from envs.env_variables import EnvVariables
from execution.execution_result import ExecutionResult
from log import Log
from utilities import PREFIX_DIR_MODELS_SAVE


class AgentStub(AbstractAgent):
    def __init__(
        self,
        algo_name: str = "ppo2",
        env_name: str = "CartPole-v1",
        tb_log_name: str = "ppo2",
        model_to_load: str = None,
        continue_learning: bool = False,
        continue_learning_suffix: str = "continue_learning",
        env_variables: EnvVariables = None,
    ):
        self.algo_name = algo_name
        self.env_name = env_name
        self.tb_log_name = tb_log_name
        self.model_to_load = model_to_load
        self.continue_learning = continue_learning
        self.continue_learning_suffix = continue_learning_suffix
        self.env_variables = env_variables
        self.logger = Log("AgentStub")

        self.best_model_to_load_path = PREFIX_DIR_MODELS_SAVE + '/' + self.algo_name + '/logs_' + self.tb_log_name

    def train(
        self,
        seed: int,
        communication_queue: Queue = None,
        current_iteration: int = -1,
        search_suffix: str = "1",
        env_variables: EnvVariables = None,
        random_search: bool = False
    ):

        continue_learning_model_save_path = (
            self.best_model_to_load_path + "_" + self.continue_learning_suffix + "_" + search_suffix + "/"
        )
        abs_continue_learning_model_save_path = os.path.abspath(continue_learning_model_save_path)
        os.makedirs(abs_continue_learning_model_save_path, exist_ok=True)

        fake_n_timesteps = 2
        return_value = False
        while fake_n_timesteps > 0:
            sleep_time = random.uniform(1.0, 2.0)
            time.sleep(sleep_time)

            if random.uniform(0.0, 1.0) < 0.5:
                return_value = True
            else:
                return_value = False

            communication_queue.put(ExecutionResult(adequate_performance=return_value))
            fake_n_timesteps -= 1

            if return_value:
                self.logger.debug("Stopping training because performance threshold was reached")
                break

        communication_queue.put(ExecutionResult(adequate_performance=return_value, task_completed=True))

    def test_with_callback(self, seed, env_variables: EnvVariables, n_eval_episodes: int = None) -> EnvPredicatePair:
        sleep_time = random.uniform(1.0, 2.0)
        time.sleep(sleep_time)

        if random.uniform(0.0, 1.0) < 0.5:
            return_value = True
        else:
            return_value = False

        return EnvPredicatePair(env_variables=env_variables, predicate=return_value)

    def test_without_callback(self, seed, n_eval_episodes: int = 0, model_path: str = None) -> Tuple[float, float]:
        return 0.0, 0.0
