from abc import ABC, abstractmethod
from queue import Queue
from typing import Dict, Tuple

from algo.env_predicate_pair import EnvPredicatePair
from envs.env_variables import EnvVariables


class AbstractAgent(ABC):
    @abstractmethod
    def train(
        self,
        seed: int,
        communication_queue: Queue = None,
        current_iteration: int = -1,
        search_suffix: str = "1",
        env_variables: EnvVariables = None,
        random_search: bool = False,
    ):
        pass

    @abstractmethod
    def test_with_callback(self, seed: int, env_variables: EnvVariables, n_eval_episodes: int = None) -> EnvPredicatePair:
        pass

    @abstractmethod
    def test_without_callback(self, seed: int, n_eval_episodes: int = 0, model_path: str = None) -> Tuple[float, float]:
        pass
