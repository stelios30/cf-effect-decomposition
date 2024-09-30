from typing import Dict, Any, List, Tuple, Union

from ced.tools.noise import NoiseModel


class CausalActor:
    def __init__(self, id: int):
        self.id = id

    def policy(self, state: Dict[str, Any]) -> List[float]:
        """
            Implements player's (stochastic) decision making policy. The order of probabilities
            is always the same. The total order is applied when sampling from this distribution
            in the `action` method below.
            OUTPUT: action probability distribution (over all available actions)
        """
        raise NotImplementedError

    def action(self, state: Dict[str, Any], act_noise: NoiseModel, return_probs: bool = False) -> Union[int, Tuple[int, List[float]]]:
        """
            Selects action based on provided state and action noise. Individual actors can specify
            their own action total order which must be honored by this method.
            OUTPUT: selected action by this agent and (optionally) the probability distribution
        """
        raise NotImplementedError

    def _get_obs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
            Extracts agent-relevant information from the state. This is always a deterministic function of
            the current state and is similar to the agent's info state in Dec-POMDP.
        """
        raise NotImplementedError()
