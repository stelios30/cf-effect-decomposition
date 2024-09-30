import copy
import numpy as np

from numpy.random import Generator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from ced.actors.causal import CausalActor
from ced.tools.noise import NoiseModel, NullNoiseModel


@dataclass
class Trajectory:
    id: Optional[int] = field(
        default=None,
        metadata={"help": "Specifies optional trajectory id."})
    states: List[Dict[str, Any]] = field(
        default_factory=list,
        metadata={"help": "Specifies states at each time-step."})
    actions: List[Union[int, Any]] = field(
        default_factory=list,
        metadata={"help": "Specifies actions at each time-step."})
    rewards: List[float] = field(
        default_factory=list,
        metadata={"help": "Specifies rewards at each time-step. Reward at r_i is obtained after taking action a_i in state s_i."})
    probs: List[List[List[float]]] = field(
        default_factory=list,
        metadata={"help": "Specifies action probabilities at each time-step, for each agent, for each action."})
    env_noise: List[Dict[str, List[NoiseModel]]] = field(
        default_factory=list,
        metadata={"help": "Specifies sampled environment noise at each time-step."})
    act_noise: List[List[NoiseModel]] = field(
        default_factory=list,
        metadata={"help": "Specifies sampled action noise at each time-step, for each agent."})

    @property
    def horizon(self) -> int:
        return len(self.states)

    def render(self) -> str:
        return f"Trajectory: {self.id}"

    def outcome(self) -> float:
        raise NotImplementedError

    def total_reward(self, discount: float = 0.99) -> float:
        rewards = np.array([r for r in self.rewards if r is not None])
        discounts = discount ** np.arange(rewards.size, dtype=np.float32)
        return np.sum(rewards * discounts).item()

class CausalEnv:
    def __init__(self, act_noise_model: NoiseModel = NullNoiseModel(), env_noise_model: NoiseModel = NullNoiseModel()):
        self.num_agents = None                 # number of controllable agents
        self.horizon = None                    # environment horizon
        self.act_noise_model = act_noise_model # specifies noise model of the agent's dynamics
        self.env_noise_model = env_noise_model # specifies noise model of the environment's dynamics

    def reset(self, rng: Optional[Generator] = None, env_noise: Optional[Dict[str, List[NoiseModel]]] = None):
        """
            Performs an environment reset.
            Input: optional random number generator
            Output: initial state
        """
        raise NotImplementedError

    def step(self, state: Dict[str, Any], actions: Tuple[int, ...], env_noise: Dict[str, List[NoiseModel]], rng: Optional[Generator] = None):
        """
            Performs a single environment step.
            Input: current state, agent's actions, sampled environment noise
            Output: next state, reward
        """
        raise NotImplementedError

    def sample_trajectory(
        self,
        agents: List[CausalActor],
        act_noise: Optional[List[List[NoiseModel]]] = None,
        env_noise: Optional[List[Dict[str, List[NoiseModel]]]] = None,
        initial_state: Dict[str, Any] = None,
        do_operators: Optional[List[Dict[int, int]]] = None,
        rng: Optional[Generator] = None,
        horizon: Optional[int] = None,
        pad: bool = False,
    ) -> Trajectory:
        """
            Samples a trajectory following environment's dynamics. If a specific noise is not provided, it is randomly generated.
            Note that we have one noise entity per each time-step. The do-operator is defined as an action override, where for
            each time-step we have a dictionary mapping agent's id to the action that it should do. The optional horizon parameter
            specifies the maximum length of the trajectory. If omitted, we use the environment's default horizon. The pad flag
            indicates if trajectories that terminate before the horizon should be padded with null states and actions.
        """
        trajectory = Trajectory()
        horizon = horizon if horizon is not None else self.horizon

        # sample the initial state
        curr_env_noise = env_noise[0] if env_noise and env_noise[0] is not None else self._get_env_noise(rng=rng)
        state = copy.deepcopy(initial_state) if initial_state is not None else self.reset(rng=rng, env_noise=curr_env_noise)
        reward = 0.0

        # sample the trajectory
        for t in range(horizon - 1):
            # breaks the loop if the state is terminal or pads the trajectory with terminal states
            # if it terminated before the given horizon
            if not pad and getattr(state, "terminal", False):
                break
            elif not pad and isinstance(state, dict) and state.get("terminal", False):
                break
            elif pad and (getattr(state, "terminal", False) or isinstance(state, dict) and state.get("terminal", False)):
                trajectory.states.append(state)
                trajectory.env_noise.append(None)
                trajectory.act_noise.append(None)
                trajectory.actions.append([None] * self.num_agents)
                trajectory.probs.append([None] * self.num_agents)
                trajectory.rewards.append(None)
                state = copy.deepcopy(state)
                continue

            # fetches action noise
            if act_noise and t < len(act_noise) and act_noise[t] is not None:
                # do not generate action noise, if it was explicitly passed
                curr_act_noise = act_noise[t]
            else:
                # sample new action noise
                curr_act_noise = self._get_act_noise(agents=agents, state=state, rng=rng)

            # appends new information to trajectory
            trajectory.states.append(state)
            trajectory.rewards.append(reward)
            trajectory.env_noise.append(curr_env_noise)
            trajectory.act_noise.append(curr_act_noise)

            # append taken action (a_t = f(s_t, u_^a_t))
            do_operator = do_operators[t] if do_operators is not None and t < len(do_operators) else None
            actions, probs = self._get_act(state=state, agents=agents, act_noise=curr_act_noise, do_operator=do_operator)
            trajectory.actions.append(actions)
            trajectory.probs.append(probs)

            # fetches environment noise
            if env_noise and t + 1 < len(env_noise) and env_noise[t + 1] is not None:
                # do not generate environment noise, if it is explicitly passed
                next_env_noise = env_noise[t + 1]
            else:
                # sample noise for the current state
                next_env_noise = self._get_env_noise(rng=rng)

            # transitions to next state (s_t = f(s_t-1, a_t-1, u_t))
            next_state, next_reward = self.step(state, actions, next_env_noise, rng)

            # advance to the next state
            state, curr_env_noise, reward = next_state, next_env_noise, next_reward

        # append final state
        trajectory.states.append(state)
        trajectory.rewards.append(reward)
        trajectory.actions.append([None] * self.num_agents)
        trajectory.probs.append([None] * self.num_agents)

        return trajectory

    def sample_noise_from_posterior(self, trajectory: Trajectory, agents: List[CausalActor], rng: Optional[Generator] = None) -> Tuple[List[List[NoiseModel]], List[Dict[str, NoiseModel]]]:
        """
            For a given trajectory, samples environment and agent noise that could have realised that trajectory.
        """
        act_noise, env_noise = [None] * trajectory.horizon, [None] * trajectory.horizon

        for t in range(trajectory.horizon):
            if t < trajectory.horizon - 1:
                act_noise[t] = self._get_act_noise(time_step=t, trajectory=trajectory, rng=rng, agents=agents)
            env_noise[t] = self._get_env_noise(time_step=t, trajectory=trajectory, rng=rng)

        return act_noise, env_noise

    def get_available_actions(self, state: Dict[str, Any], agent_id: int) -> List[int]:
        """
            Returns list of actions available to the specified agent in a given state.
        """
        raise NotImplementedError

    def _get_act(self, state: Dict[str, Any], agents: List[CausalActor], act_noise: List[NoiseModel], do_operator: Optional[Dict[int, int]] = None) -> Tuple[List[int], List[float]]:
        """
            Returns agent's action in a given state, based on the passed action noise, one for each acting agent. For all other
            non-acting agents, we set their actions to `None`.
        """
        actions = [None] * self.num_agents
        probs = [None] * self.num_agents

        for agent in self._get_acting_agents(state, agents):
            if do_operator is not None and do_operator.get(agent.id, None) is not None:
                actions[agent.id] = do_operator[agent.id]
            elif act_noise[agent.id] is not None:
                actions[agent.id], probs[agent.id] = agent.action(state, act_noise[agent.id], return_probs=True)
            else:
                raise RuntimeError(f"Action noise missing for agent {agent.id} in state {state}")

        return actions, probs

    def _get_acting_agents(self, state: Dict[str, Any], agents: List[CausalActor]) -> List[CausalActor]:
        """
            Returns a list of agents which can act in the given state.
        """
        raise NotImplementedError

    def _get_act_noise(self, agents: List[CausalActor], state: Optional[Dict[str, Any]] = None, trajectory: Optional[Trajectory] = None, time_step: Optional[int] = None, rng: Optional[Generator] = None) -> List[NoiseModel]:
        """
            Samples action noise for each of the agents that can act in the current state. The output is list
            of noise elements, one for each agent. If a `trajectory` is passed, we sample the noise from the
            posterior of that trajectory.
        """
        assert state or (trajectory and time_step is not None), "Either state (for regular) or trajectory and time_step (for posterior) sampling should be passed"

        # perform regular noise sampling, when no trajectory is passed
        if not trajectory:
            return [self.act_noise_model.sample(rng=rng) for _ in agents]

        # posterior sampling for a given trajectory
        probs = trajectory.probs[time_step]
        realised = trajectory.actions[time_step]

        noise = [self.act_noise_model.sample_posterior(probs=prob, realised=act, rng=rng) for prob, act in zip(probs, realised)]
        assert realised == [n.choice(p, rng=rng) for p, n in zip(probs, noise)]

        return noise

    def _get_env_noise(self, trajectory: Optional[Trajectory] = None, time_step: Optional[int] = None, rng: Optional[Generator] = None) -> Dict[str, NoiseModel]:
        """
            Environment noise is used to govern the environment's dynamics. To handle a wider range of cases
            we allow the noise to be grouped under a dictionary, for different environment elements. If a
            `trajectory` is passed, we sample the noise from the posterior of that trajectory.
        """
        return {}
