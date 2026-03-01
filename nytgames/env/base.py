from abc import ABC, abstractmethod
import gymnasium as gym


class NYTGameEnv(gym.Env, ABC):
    """Abstract base class for all NYT game environments.

    Subclasses must:
    - Pass a config object to super().__init__(config)
    - Implement _get_obs, _get_reward, _is_truncated, and _is_terminated
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def _get_obs(self) -> dict: ...

    @abstractmethod
    def _get_reward(self, action) -> float: ...

    @abstractmethod
    def _is_truncated(self) -> bool: ...

    @abstractmethod
    def _is_terminated(self) -> bool: ...
