import pickle
import numpy as np

from pathlib import Path
from typing import List, Union, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field, fields

from ced.tools.noise import NoiseModel
from ced.actors.causal import CausalActor
if TYPE_CHECKING:
    from ced.envs.sepsis import State


@dataclass
class SepsisAction:
    ANTIBIOTIC = 0
    VENTILATION = 1
    VASOPRESSORS = 2
    NUM_TOTAL = 2 * 2 * 2    # 8 actions for the sepsis environment
    NUM_FULL = 2 * 2 * 2 + 1 # 8 actions for the sepsis environment and 1 no-op action for the clinician

    antibiotic: int = field(
        default=0,
        metadata={"code": "A", "help": "Indicates whether the antibiotic is administered."})
    ventilation: int = field(
        default=0,
        metadata={"code": "E", "help": "Indicates whether ventilation is used."})
    vasopressors: int = field(
        default=0,
        metadata={"code": "V", "help": "Indicates whether vasopressors is administered."})

    @classmethod
    def from_index(cls, index: int) -> "SepsisAction":
        assert 0 <= index < cls.NUM_TOTAL, f"Index must be integer in [0, {cls.NUM_TOTAL})"

        # assumes order (antibiotic, ventilation, vasopressors), see property 'index'
        curr_index = index
        curr_base = cls.NUM_TOTAL / 2

        # decode if antibiotics are administered
        antibiotic = np.floor(curr_index / curr_base).astype(int)
        curr_index %= curr_base
        curr_base /= 2

        # decode if ventilation is used
        ventilation = np.floor(curr_index / curr_base).astype(int)
        curr_index %= curr_base
        curr_base /= 2

        # decode if vasopressors are administered
        vasopressors = np.floor(curr_index / curr_base).astype(int)

        return cls(antibiotic, ventilation, vasopressors)

    @property
    def index(self) -> int:
        return 4 * self.antibiotic + 2 * self.ventilation + self.vasopressors

    def __hash__(self) -> int:
        return self.index

    def __str__(self) -> str:
        result = ""

        antibiotic, ventilation, vasopressors = fields(self)
        antibiotic, ventilation, vasopressors = antibiotic.metadata["code"], ventilation.metadata["code"], vasopressors.metadata["code"]

        result = f"{result}{antibiotic}" if self.antibiotic else result
        result = f"{result}{ventilation}" if self.ventilation else result
        result = f"{result}{vasopressors}" if self.vasopressors else result

        return result or "-"


class SepsisActor(CausalActor):
    def __init__(self, id: int, policy: Optional[Union[np.ndarray, str, Path]] = None, rng: Optional[np.random.Generator] = None):
        super().__init__(id)
        self.policy_map = policy
        self.rng = rng
        if isinstance(self.policy_map, str) or isinstance(self.policy_map, Path):
            with open(self.policy_map, "rb") as f: self.policy_map = pickle.load(f)

    def policy(self, state: "State") -> List[float]:
        # return random action, if no policy is provided
        if self.policy_map is None:
            return [1.0 / SepsisAction.NUM_TOTAL] * SepsisAction.NUM_TOTAL + [0.0]

        # choose probabilities according to policy
        # probs is one-hot vector, because policy is deterministic
        probs = self.policy_map[state.index]

        return probs.tolist() + [0.0]

    def action(self, state: "State", act_noise: NoiseModel, return_probs: bool = False) -> Union[int, Tuple[int, List[float]]]:
        probs = self.policy(state)
        acts = np.arange(SepsisAction.NUM_FULL).tolist()

        action = act_noise.choice(probs, self.rng)
        action = acts[action]

        return action if not return_probs else (action, probs)


class AIActor(SepsisActor):
    def __init__(self, id: int, policy: Union[np.ndarray, str, Path], rng: Optional[np.random.Generator] = None):
        super().__init__(id, policy=policy, rng=rng)


class ClinicianActor(SepsisActor):
    def __init__(self, id: int, policy: Union[np.ndarray, str, Path], trust: float = 0.0, rng: Optional[np.random.Generator] = None):
        super().__init__(id=id, policy=policy, rng=rng)
        self.trust = trust

    def policy(self, state: "State") -> List[float]:
        act_cl = self.policy_map[state.index].argmax()

        probs = np.zeros(SepsisAction.NUM_FULL)
        probs[-1] = self.trust # do not intervene with probability "trust"
        probs[act_cl] = 1 - self.trust

        return probs
