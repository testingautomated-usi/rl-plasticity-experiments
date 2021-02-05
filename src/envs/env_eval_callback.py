from abc import ABC, abstractmethod
from typing import Dict, Tuple


class EnvEvalCallback(ABC):
    @abstractmethod
    def evaluate_env(self, model, env, n_eval_episodes, sb_version='sb2') -> Tuple[bool, Dict]:
        pass

    @abstractmethod
    def get_reward_threshold(self) -> bool:
        pass
