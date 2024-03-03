from typing import Any, List, Mapping, Tuple, Union


class RewardFunction:
    """
    Base reward function class.
    """

    def calculate(self, observations: Mapping[str, Union[int, float]]) -> float:
        """Calculates reward.

        Parameters
        ----------
        observations: Mapping[str, Union[int, float]]
            dictionary of the observations

        Returns
        -------
        reward: float
            Reward for transition to current timestep.
        """

        ###################################################################
        #####                Specify your reward here                 #####
        ###################################################################
        reward=0

        return reward