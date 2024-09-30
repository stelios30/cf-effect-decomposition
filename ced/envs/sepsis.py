# The original single-agent sepsis simulator was implemented by:
#   Michael Oberst, David Sontag; Counterfactual Off-Policy Evaluation with Gumbel-Max Structural Causal Models
#   https://github.com/clinicalml/gumbel-max-scm
# Implemented here is a new multi-agent version of the sepsis simulator, involving an AI and a clinician agent.

import json
import pickle
import numpy as np

from pathlib import Path
from numpy.random import Generator
from typing import List, Literal, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field, fields, replace

from ced.envs.causal import CausalEnv, Trajectory
from ced.actors.sepsis import SepsisActor, SepsisAction
from ced.tools.utils import get_probability_ranges
from ced.tools.noise import NoiseModel, NullNoiseModel, SepsisStateUniformNoiseModel


@dataclass
class State:
    NUM_TOTAL = 3 * 3 * 2 * 5 * 2 * 2 * 2 * 2

    diabetes: Literal[0, 1] = field(
        default=0,
        metadata={"cardinality": 2, "help": "Indicates whether the patient has diabetes."})
    heart_rate: Literal[0, 1, 2] = field(
        default=0,
        metadata={"cardinality": 3, "help": "Patients heart rate; 0: low, 1: normal, 2: high"})
    sys_bp: Literal[0, 1, 2] = field(
        default=0,
        metadata={"cardinality": 3, "help": "Patients systolic blood pressure; 0: low, 1: normal, 2: high"})
    per_oxygen: Literal[0, 1] = field(
        default=0,
        metadata={"cardinality": 2, "help": "Patients percent of oxygen saturation; 0: low, 1: normal"})
    glucose: Literal[0, 1, 2, 3, 4] = field(
        default=0,
        metadata={"cardinality": 5, "help": "Patients glucose level; 0: very low, 1: low, 2: normal, 3: high, 4: very high"})
    antibiotics: Literal[0, 1] = field(
        default=0,
        metadata={"cardinality": 2, "help": "Indicates whether the antibiotic is administered."})
    vasopressors: Literal[0, 1] = field(
        default=0,
        metadata={"cardinality": 2, "help": "Indicates whether vasopressors are administered."})
    ventilation: Literal[0, 1] = field(
        default=0,
        metadata={"cardinality": 2, "help": "Indicates whether ventilation is used."})
    player: Literal[0, 1] = field(
        default=0,
        metadata={"help": "Indicates which player is acting; 0: AI, 1: clinician"})
    act_ai: Literal[0, 1, 2, 3, 4, 5, 6, 7] = field(
        default=None,
        metadata={"help": "Indicates which action the AI would have taken in this state. Used in clinician's policy to determine if the override will happen."})

    @classmethod
    def from_index(cls, index: int) -> "State":
        assert 0 <= index < cls.NUM_TOTAL, f"Index must be integer in [0, {cls.NUM_TOTAL})"

        curr_index = index
        curr_items = [] # collects extracted properties in the order of their definition above

        # decodes if diabetes is present
        curr_base = cls.NUM_TOTAL / 2
        curr_items.append(np.floor(curr_index / curr_base).astype(int))
        curr_index %= curr_base

        # decodes heart rate value
        curr_base /= 3
        curr_items.append(np.floor(curr_index / curr_base).astype(int))
        curr_index %= curr_base

        # decodes systolic blood pressure value
        curr_base /= 3
        curr_items.append(np.floor(curr_index / curr_base).astype(int))
        curr_index %= curr_base

        # decodes percent of oxygen saturation value
        curr_base /= 2
        curr_items.append(np.floor(curr_index / curr_base).astype(int))
        curr_index %= curr_base

        # decodes glucose level value
        curr_base /= 5
        curr_items.append(np.floor(curr_index / curr_base).astype(int))
        curr_index %= curr_base

        # decodes antibiotic value
        curr_base /= 2
        curr_items.append(np.floor(curr_index / curr_base).astype(int))
        curr_index %= curr_base

        # decodes vasopressors value
        curr_base /= 2
        curr_items.append(np.floor(curr_index / curr_base).astype(int))
        curr_index %= curr_base

        # decodes ventilation value
        curr_base /= 2
        curr_items.append(np.floor(curr_index / curr_base).astype(int))

        return cls(*curr_items)

    @property
    def index(self) -> int:
        names = ["diabetes", "heart_rate", "sys_bp", "per_oxygen", "glucose", "antibiotics", "vasopressors", "ventilation"]
        values = [getattr(self, f.name) for f in fields(self) if f.name in names]
        counts = [f.metadata["cardinality"] for f in fields(self) if f.name in names]

        index = 0
        curr_base = 1

        for i in reversed(range(len(values))):
            index += curr_base * values[i]
            curr_base *= counts[i]

        return index

    @property
    def terminal(self) -> bool:
        return self.patient_dead or self.patient_discharged

    @property
    def reward(self) -> float:
        if self.patient_dead:
            return -1.0
        elif self.patient_discharged:
            return +1.0
        else:
            return 0.0

    @property
    def patient_discharged(self) -> bool:
        return self.num_abnormal == 0 and not self.on_treatment

    @property
    def patient_dead(self) -> bool:
        return self.num_abnormal >= 3

    @property
    def num_abnormal(self) -> int:
        result = 0
        result += 1 if self.heart_rate != 1 else 0
        result += 1 if self.sys_bp != 1 else 0
        result += 1 if self.per_oxygen != 1 else 0
        result += 1 if self.glucose != 2 else 0
        return result

    @property
    def on_treatment(self) -> bool:
        return self.antibiotics == 1 or self.vasopressors == 1 or self.ventilation == 1

    def __equal__(self, other: object) -> bool:
        return self.index == other.index if isinstance(other, State) else False

    def __hash__(self) -> int:
        return self.index


@dataclass
class SepsisTrajectory(Trajectory):
    states: List[State] = field(
        default_factory=list,
        metadata={"help": "Specifies states at each time-step."})

    def render(self) -> str:
        result = f"Trajectory: {self.id}\n"

        for i, (state, action) in enumerate(zip(self.states, self.actions)):
            result += f"    Time-step {i}; Actions (AI, CL) {action}; Reward {state.reward}; State=(glucose={state.glucose}, heart_rate={state.heart_rate}, sys_bp={state.sys_bp}, per_oxygen={state.per_oxygen}, diabetes={state.diabetes})\n"

        return result

    def outcome(self) -> float:
        """ returns 1 if patient survived and 0 if not
        """
        return not self.states[-1].patient_dead


class Sepsis(CausalEnv):
    def __init__(
        self,
        transition_probabilities: Optional[Union[str, Path, Dict[str, float]]] = None,
        dynamics: Optional[Dict[str, np.ndarray]] = None,
        max_horizon: int = 20,
        act_noise_model: NoiseModel = NullNoiseModel(),
        env_noise_model: SepsisStateUniformNoiseModel = SepsisStateUniformNoiseModel(),
        turn_based: bool = False,
    ):
        super().__init__(act_noise_model=act_noise_model, env_noise_model=env_noise_model)

        assert transition_probabilities is not None or dynamics is not None, "Either transition probabilities or dynamics must be specified."

        self.horizon = max_horizon
        self.num_agents = 2 if turn_based else 1
        self.probs = transition_probabilities
        self.dynamics = dynamics
        self.turn_based = turn_based

        if isinstance(self.probs, Path) or isinstance(self.probs, str):
            with open(self.probs, "r") as f:
                self.probs = json.load(f)

        if isinstance(self.dynamics, Path) or isinstance(self.dynamics, str):
            with open(self.dynamics, "rb") as f:
                self.dynamics = pickle.load(f)

        assert "transition_matrix" in self.dynamics if self.dynamics is not None else True, "Transition matrix must be specified."
        assert "initial_state_distribution" in self.dynamics if self.dynamics is not None else True, "Initial state distribution must be specified."

        if self.dynamics and env_noise_model.order is not None:
            # applies total order to transition matrix and initial-state distribution, to reduce the computation time
            tm = self.dynamics["transition_matrix"]
            insd = self.dynamics["initial_state_distribution"]

            # for each (action, state) pair, swaps the probabilities of successor (or initial) state to match the total order
            tm_ordered = np.empty((SepsisAction.NUM_TOTAL, State.NUM_TOTAL, 0), dtype=np.float64)
            insd_ordered = []
            for new_index in range(State.NUM_TOTAL):
                old_index = env_noise_model.order.decode_index[new_index]
                tm_ordered = np.concatenate((tm_ordered, tm[:, :, old_index][:, :, None]), axis=-1)
                insd_ordered += [insd[old_index]]

            # for each (action, state) pair, computes the probability ranges of the successor state
            tm_ranges = [[] for _ in range(SepsisAction.NUM_TOTAL)]
            for a in range(SepsisAction.NUM_TOTAL):
                for s in range(State.NUM_TOTAL):
                    probs = tm_ordered[a, s, :].tolist()
                    tm_ranges[a].append(get_probability_ranges(probs))

            # preserves probability ranges, to be used in the `step` method
            self.dynamics["transition_matrix"] = tm_ordered
            self.dynamics["transition_matrix_ranges"] = tm_ranges
            self.dynamics["initial_state_distribution"] = insd_ordered
            self.dynamics["initial_state_distribution_ranges"] = get_probability_ranges(insd_ordered)

    def reset(self, rng: Optional[Generator] = None, env_noise: Optional[Dict[str, List[NoiseModel]]] = None) -> State:
        # if rng is not specified, create a new one with random seed
        rng = rng or np.random.default_rng()

        # if dynamics are specified, sample initial state from the distribution
        if self.dynamics:
            ranges = self.dynamics["initial_state_distribution_ranges"]
            state = State.from_index(env_noise["state"].choice(ranges))
            return replace(state, player=0) if self.turn_based else state

        # without dynamics, generate random initial state using specified initial probabilities
        diabetic = rng.binomial(1, self.probs["initial"]["diabetes"])
        heart_rate = rng.choice(np.arange(3), p=np.array(self.probs["initial"]["heart_rate"]))
        sys_bp = rng.choice(np.arange(3), p=np.array(self.probs["initial"]["sys_bp"]))
        per_oxygen = rng.choice(np.arange(2), p=np.array(self.probs["initial"]["per_oxygen"]))
        glucose = rng.choice(np.arange(5), p=np.array(self.probs["initial"]["glucose_diabetic"] if diabetic else self.probs["initial"]["glucose"]))
        state = State(
            diabetes=diabetic, heart_rate=heart_rate, sys_bp=sys_bp,
            per_oxygen=per_oxygen, glucose=glucose, antibiotics=0,
            vasopressors=0, ventilation=0, player=0)

        # ensure sampled state is not terminal
        return state if not state.terminal else self.reset(rng=rng)

    def step(self, state: Union[State, int], actions: Tuple[int, ...], env_noise: Optional[Dict[str, NoiseModel]] = None, rng: Optional[Generator] = None) -> Tuple[State, float]:
        # if rng is not specified, create a new one with random seed
        rng = rng or np.random.default_rng()

        # decode state indices into objects
        state = State.from_index(state) if isinstance(state, int) else state

        # decode action indices into object
        if not self.turn_based:
            # if only one agent is present, directly decode its action
            action = SepsisAction.from_index(actions[0])
        elif state.player == 0:
            # if AI agent is playing, update state with its action and switch to clinician
            return replace(state, player=1 - state.player, act_ai=actions[state.player]), state.reward
        elif state.player == 1:
            # if clinician is playing, handle the case where it does not intervene (action = 8)
            action = actions[state.player] if actions[state.player] < SepsisAction.NUM_TOTAL else state.act_ai
            action = SepsisAction.from_index(action)

        # if dynamics are specified, sample next state from the transition matrix using the noise model
        if self.dynamics:
            ranges = self.dynamics["transition_matrix_ranges"][action.index][state.index]
            next_state = State.from_index(env_noise["state"].choice(ranges))
            next_state.player = 1 - state.player if self.turn_based else 0
            return next_state, next_state.reward

        # if dynamics are not specified, manually transition to the successor state
        next_state = replace(state, player=1 - state.player if self.turn_based else 0, act_ai=None)
        heart_rate_fluct, sys_bp_fluct, per_oxygen_fluct, glucose_fluct = False, False, False, True

        if action.antibiotic == 1:
            # prescribes antibiotics to patient
            next_state = self._transition_antibiotics_on(next_state, rng=rng)
            heart_rate_fluct = sys_bp_fluct = False
        elif state.antibiotics == 1:
            # remove patient from antibiotics
            next_state = self._transition_antibiotics_off(next_state, rng=rng)
            heart_rate_fluct = sys_bp_fluct = False
        else:
            # antibiotics are not prescribed, patient is subject to fluctuations
            heart_rate_fluct = sys_bp_fluct = True

        if action.ventilation == 1:
            # use ventilation to treat patient
            next_state = self._transition_ventilation_on(next_state, rng=rng)
            per_oxygen_fluct = False
        elif state.ventilation == 1:
            # remove patient from ventilation
            next_state = self._transition_ventilation_off(next_state, rng=rng)
            per_oxygen_fluct = False
        else:
            # ventilation is not used, patient is subject to fluctuations
            per_oxygen_fluct = True

        if action.vasopressors == 1:
            # prescribes vasopressors to patient
            next_state = self._transition_vasopressors_on(next_state, rng=rng)
            sys_bp_fluct = glucose_fluct = False
        elif state.vasopressors == 1:
            # remove patient from vasopressors
            next_state = self._transition_vasopressors_off(next_state, rng=rng)
            sys_bp_fluct = False

        # incorporate random fluctuations
        next_state = self._transition_fluctuations(next_state, heart_rate_fluct, sys_bp_fluct, per_oxygen_fluct, glucose_fluct, rng=rng)

        return next_state, next_state.reward

    def sample_trajectory(
        self,
        agents: List[SepsisActor],
        act_noise: Optional[List[List[NoiseModel]]] = None,
        env_noise: Optional[List[Dict[str, List[NoiseModel]]]] = None,
        initial_state: Union[State, int] = None,
        do_operators: Optional[List[Dict[int, int]]] = None,
        rng: Optional[Generator] = None,
        horizon: Optional[int] = None,
        pad: bool = False,
    ) -> SepsisTrajectory:
        initial_state = State.from_index(initial_state) if isinstance(initial_state, int) else initial_state
        t = super().sample_trajectory(agents, act_noise, env_noise, initial_state, do_operators, rng, horizon, pad)
        t = SepsisTrajectory(id=t.id, states=t.states, actions=t.actions, rewards=t.rewards, probs=t.probs, env_noise=t.env_noise, act_noise=t.act_noise)
        return t

    def get_available_actions(self, state: State, agent_id: int) -> List[int]:
        if agent_id == 0:
            return [] if state.terminal else [i for i in range(SepsisAction.NUM_TOTAL)]
        elif agent_id == 1:
            return [] if state.terminal else [i for i in range(SepsisAction.NUM_FULL)]
        else:
            raise RuntimeError(f"Invalid agent id {agent_id}")

    def _get_acting_agents(self, state: State, agents: List[SepsisActor]) -> List[SepsisActor]:
        return [agents[state.player]] if self.turn_based else agents

    def _transition_antibiotics_on(self, state: State, rng: Optional[Generator] = None) -> State:
        """prescribes antibiotics to patient
           heart rate:
               from high to normal w.p. 0.5
           systolic blood pressure:
               from high to normal w.p. 0.5
        """
        state.antibiotics = 1
        if state.heart_rate == 2 and rng.uniform(0, 1) < self.probs["antibiotics"]["on"]["heart_rate_high_to_normal"]:
            state.heart_rate = 1
        if state.sys_bp == 2 and rng.uniform(0, 1) < self.probs["antibiotics"]["on"]["sys_bp_high_to_normal"]:
            state.sys_bp = 1
        return state

    def _transition_antibiotics_off(self, state: State, rng: Optional[Generator] = None) -> State:
        """removes patient from antibiotics
           heart rate:
               from normal to high w.p. 0.1
           systolic blood pressure:
               from normal to high w.p. 0.1
        """
        state.antibiotics = 0
        if state.heart_rate == 1 and rng.uniform(0, 1) < self.probs["antibiotics"]["off"]["heart_rate_normal_to_high"]:
            state.heart_rate = 2
        if state.sys_bp == 1 and rng.uniform(0, 1) < self.probs["antibiotics"]["off"]["sys_bp_normal_to_high"]:
            state.sys_bp = 2
        return state

    def _transition_ventilation_on(self, state: State, rng: Optional[Generator] = None) -> State:
        """use ventilation to treat patient
           percentage of oxygen saturation:
               from low to normal w.p. 0.7
        """
        state.ventilation = 1
        if state.per_oxygen == 0 and rng.uniform(0, 1) < self.probs["ventilation"]["on"]["per_oxygen_low_to_normal"]:
            state.per_oxygen = 1
        return state

    def _transition_ventilation_off(self, state: State, rng: Optional[Generator] = None) -> State:
        """remove patient from ventilation
           percentage of oxygen saturation:
               from normal to low w.p. 0.1
        """
        state.ventilation = 0
        if state.per_oxygen == 1 and rng.uniform(0, 1) < self.probs["ventilation"]["off"]["per_oxygen_normal_to_low"]:
            state.per_oxygen = 0
        return state

    def _transition_vasopressors_on(self, state: State, rng: Optional[Generator] = None) -> State:
        """prescribes vasopressors to patient
           for non-diabetic patients:
               systolic blood pressure:
                   from low to normal  w.p. 0.7
                   from normal to high w.p. 0.7
           for diabetic patients:
               systolic blood pressure:
                   from low    to normal w.p. 0.5
                   from low    to high   w.p. 0.4
                   from normal to high   w.p. 0.9,
               glucose:
                   raise by 1 w.p. 0.5
        """
        state.vasopressors = 1
        probs = self.probs["vasopressors"]["on"]

        # handles non-diabetic patients
        if state.diabetes == 0:
            if state.sys_bp == 0 and rng.uniform(0, 1) < probs["non-diabetic"]["sys_bp_low_to_normal"]:
                state.sys_bp = 1
            if state.sys_bp == 1 and rng.uniform(0, 1) < probs["non-diabetic"]["sys_bp_normal_to_high"]:
                state.sys_bp = 2
            return state

        # handles diabetic patients
        if state.sys_bp == 1:
            if rng.uniform(0, 1) < probs["diabetic"]["sys_bp_normal_to_high"]:
                state.sys_bp = 2
        elif state.sys_bp == 0:
            up_prob = rng.uniform(0, 1)
            if up_prob < probs["diabetic"]["sys_bp_low_to_normal"]:
                state.sys_bp = 1
            elif up_prob < probs["diabetic"]["sys_bp_low_to_high"] + probs["diabetic"]["sys_bp_low_to_normal"]:
                state.sys_bp = 2

        # raises glucose level
        if rng.uniform(0, 1) < probs["diabetic"]["raise_glucose"]:
            state.glucose = min(4, state.glucose + 1)

        return state

    def _transition_vasopressors_off(self, state: State, rng: Optional[Generator] = None) -> State:
        """remove patient from vasopressors
           for non-diabetic patients:
              systolic blood pressure:
                 from high to normal w.p. 0.1
                 from normal to low w.p. 0.1
           for diabetic patients:
              systolic blood pressure:
                 from high to normal w.p. 0.05
                 from normal to low w.p. 0.05
        """
        state.vasopressors = 0
        probs = self.probs["vasopressors"]["off"]
        if state.diabetes == 0:
            if rng.uniform(0, 1) < probs["non-diabetic"]["lower_sys_bp"]:
                state.sys_bp = max(0, state.sys_bp - 1)
        else:
            if rng.uniform(0, 1) < probs["diabetic"]["lower_sys_bp"]:
                state.sys_bp = max(0, state.sys_bp - 1)
        return state

    def _transition_fluctuations(self, state: State, heart_rate_fluct: bool, sys_bp_fluct: bool, per_oxygen_fluct: bool, glucose_fluct: bool, rng: Optional[Generator] = None) -> State:
        """incorporates random fluctuations for indicated properties
           all properties fluctuate by +-1 w.p. 0.1
           glucose fluctuates by +-1 w.p. 0.3 only if patient is diabetic
        """
        if heart_rate_fluct:
            heart_rate_prob = rng.uniform(0, 1)
            if heart_rate_prob < self.probs["fluctuations"]["heart_rate"]:
                state.heart_rate = max(0, state.heart_rate - 1)
            elif heart_rate_prob < 2 * self.probs["fluctuations"]["heart_rate"]:
                state.heart_rate = min(2, state.heart_rate + 1)

        if sys_bp_fluct:
            sys_bp_prob = rng.uniform(0, 1)
            if sys_bp_prob < self.probs["fluctuations"]["sys_bp"]:
                state.sys_bp = max(0, state.sys_bp - 1)
            elif sys_bp_prob < 2 * self.probs["fluctuations"]["sys_bp"]:
                state.sys_bp = min(2, state.sys_bp + 1)

        if per_oxygen_fluct:
            per_oxygen_prob = rng.uniform(0, 1)
            if per_oxygen_prob < self.probs["fluctuations"]["per_oxygen"]:
                state.per_oxygen = max(0, state.per_oxygen - 1)
            elif per_oxygen_prob < 2 * self.probs["fluctuations"]["per_oxygen"]:
                state.per_oxygen = min(1, state.per_oxygen + 1)

        if glucose_fluct:
            glucose_prob = rng.uniform(0, 1)
            if state.diabetes == 0:
                if glucose_prob < self.probs["fluctuations"]["glucose"]["non-diabetic"]:
                    state.glucose = max(0, state.glucose - 1)
                elif glucose_prob < 2 * self.probs["fluctuations"]["glucose"]["non-diabetic"]:
                    state.glucose = min(4, state.glucose + 1)
            else:
                if glucose_prob < self.probs["fluctuations"]["glucose"]["diabetic"]:
                    state.glucose = max(0, state.glucose - 1)
                elif glucose_prob < 2 * self.probs["fluctuations"]["glucose"]["diabetic"]:
                    state.glucose = min(4, state.glucose + 1)

        return state

    def _get_act_noise(self, agents: List[SepsisActor], state: Optional[State] = None, trajectory: Optional[Trajectory] = None, time_step: Optional[int] = None, rng: Optional[Generator] = None) -> List[NoiseModel]:
        assert state or (trajectory and time_step is not None), "Either state (for regular) or trajectory and time_step (for posterior) sampling should be passed"

        # perform regular noise sampling, when no trajectory is passed
        if not trajectory:
            if self.turn_based:
                # sample one (new) noise element only for acting agent
                return [self.act_noise_model.sample(rng=rng) if state.player == a.id else None for a in agents]
            else:
                # for each agent, sample one (new) noise element, for each available action if necessary
                return [self.act_noise_model.sample(rng=rng) for _ in agents]

        # perform posterior sampling for a given trajectory
        state = trajectory.states[time_step]
        probs = trajectory.probs[time_step][state.player]
        realised = trajectory.actions[time_step][state.player]

        noise = self.act_noise_model.sample_posterior(probs=probs, realised=realised, rng=rng)
        assert realised == noise.choice(probs, rng=rng)

        result = [None] * len(agents)
        result[state.player] = noise
        return result

    def _get_env_noise(self, trajectory: Optional[Trajectory] = None, time_step: Optional[int] = None, rng: Optional[Generator] = None) -> Dict[str, NoiseModel]:
        assert (not trajectory and time_step is None) or (trajectory and time_step is not None), "Either both trajectory and time_step should be passed (posterior sampling), or none of them (regular sampling)"
        assert self.dynamics if (trajectory and time_step is not None) else True, "Posterior sampling from a given trajectory requires sepsis MDP dynamics"

        # === perform regular noise sampling, when no trajectory is passed ===
        if not trajectory:
            return {"state": self.env_noise_model.sample(rng=rng)}

        # === perform posterior sampling, when a trajectory is given ===

        # sample noise for the initial state
        if time_step == 0:
            state = trajectory.states[time_step]

            probs_ordered = self.dynamics["initial_state_distribution"]
            realised_ordered = self.env_noise_model.order.encode_index[state.index]
            noise = {"state": self.env_noise_model.sample_posterior(probs=probs_ordered, realised=realised_ordered, rng=rng)}

            assert state.index == noise["state"].choice(self.dynamics["initial_state_distribution_ranges"]), "Posterior sampling failed: realised initial state does not match the sampled one"
            return noise

        # sample noise for subsequent states
        state = trajectory.states[time_step - 1]
        state_next = trajectory.states[time_step]

        # do not sample noise for steps when AI agent acts, state transitions deterministically
        if self.turn_based and state.player == 0:
            return {"state": None}

        # retrieve action from the trajectory, handling the case when
        # clinician does not override AI's action
        action = trajectory.actions[time_step - 1][state.player]
        action = action if action < SepsisAction.NUM_TOTAL else state.act_ai

        probs_ordered = self.dynamics["transition_matrix"][action][state.index].tolist()
        realised_ordered = self.env_noise_model.order.encode_index[state_next.index]
        noise = {"state": self.env_noise_model.sample_posterior(probs=probs_ordered, realised=realised_ordered, rng=rng)}

        assert 0.0 != probs_ordered[realised_ordered], f"Posterior sampling failed: realised state is not reachable from the current one (mdp dynamics do not match the trajectory {trajectory.id})"
        assert state_next.index == noise["state"].choice(self.dynamics["transition_matrix_ranges"][action][state.index]), "Posterior sampling failed: realised value does not match the sampled one"

        return noise
