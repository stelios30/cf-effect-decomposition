from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pygame
from numpy.random import Generator

from ced.actors.causal import CausalActor
from ced.actors.grid import GridWorldAction, GridWorldActor, PlannerAction
from ced.envs.causal import CausalEnv, Trajectory
from ced.tools.noise import NoiseModel, NullNoiseModel

# Initialize environment constants
GRID_ROWS = 5
GRID_COLS = 12

NUM_AGENTS = 2

ITEM_NULL = 0
ITEM_YELLOW = 1
ITEM_GREEN = 2
ITEM_PINK = 3

REWARDS = {
    "cell": -0.2,
    "corridors": {
        "pink": {
            "goal": 180,
            "rewards": [(-30, 1/3), (-50, 1/3), (-70, 1/3)],
            "rewards_reduced": [(-5, 1/3), (-15, 1/3), (-25, 1/3)],
        },
        "green": {
            "goal": 150,
            # rewards for the green corridor are given per cell (enumerated left to right)
            "rewards_c0": [(-30, 0.3), (-40, 0.4), (-50, 0.3)],
            "rewards_c1": [(-30, 0.25), (-40, 0.5), (-50, 0.25)],
            "rewards_c2": [(-30, 0.2), (-40, 0.6), (-50, 0.2)],
            "rewards_c3": [(-30, 0.15), (-40, 0.7), (-50, 0.15)],
            "rewards_reduced_c0": [(-5, 0.3), (-10, 0.4), (-15, 0.3)],
            "rewards_reduced_c1": [(-5, 0.25), (-10, 0.5), (-15, 0.25)],
            "rewards_reduced_c2": [(-5, 0.2), (-10, 0.6), (-15, 0.2)],
            "rewards_reduced_c3": [(-5, 0.15), (-10, 0.7), (-15, 0.15)],
        },
        "yellow": {
            "goal": 90,
            "rewards": [(-25, 1/3), (-30, 1/3), (-35, 1/3)],
            "rewards_reduced": [(-2.5, 1/3), (-5, 1/3), (-7.5, 1/3)],
        },
    },
}

# Set locations of the boxes and destinations using a row-major order
LOC_BOX_1 = 0 * GRID_COLS + (GRID_COLS - 1)
LOC_BOX_2 = (GRID_ROWS - 1) * GRID_COLS + (GRID_COLS - 1)
LOC_DEST_PINK = 0 * GRID_COLS + 0
LOC_DEST_GREEN = 2 * GRID_COLS + 0
LOC_DEST_YELLOW = 4 * GRID_COLS + 0
LOC_WALLS = \
    [row * GRID_COLS + col for row in [0] for col in [4, 5, 6, 7]] + \
    [row * GRID_COLS + col for row in [4] for col in [4, 5, 6, 7]]
LOC_CORRIDORS = {
    "pink": [1 * GRID_COLS + col for col in [4, 5, 6, 7]],
    "green": [2 * GRID_COLS + col for col in [4, 5, 6, 7]],
    "yellow": [3 * GRID_COLS + col for col in [4, 5, 6, 7]],
}

class Instruction(str, Enum):
    goto_pink = "goto pink"
    goto_green = "goto green"
    goto_yellow = "goto yellow"
    examine_box_1 = "examine box 1"
    examine_box_2 = "examine box 2"
    pickup_pink = "pickup pink"
    pickup_green = "pickup green"
    pickup_yellow = "pickup yellow"

    def one_hot(self) -> np.ndarray:
        return np.array([1. if self == inst else 0. for inst in list(Instruction)])

    def get_index(self) -> int:
        return list(Instruction).index(self)

    @staticmethod
    def from_str(inst: str) -> Optional["Instruction"]:
        try:
            return Instruction(inst)
        except Exception:
            return None

    @staticmethod
    def get_sampling_weight(inst: "Instruction") -> float:
        if inst in [Instruction.goto_yellow, Instruction.goto_green, Instruction.goto_pink]:
            return 0.9 / 3
        else:
            return 0.1 / 5


@dataclass
class State:
    mode: Literal["a1", "a2", "multi-agent-w-planner"] = field(
        default="multi-agent-w-planner",
        metadata={"help": "The mode of the environment, either single-agent, multi-agent, or multi-agent with a planner."})
    stage: Literal["planning", "execution"] = field(
        default="planning",
        metadata={"help": "The stage of the environment, either planning or execution."})
    inst: Union[str, Tuple[str]] = field(
        default="",
        metadata={"help": "The instructions for each agent, or a single instruction for the single-agent mode."})
    inst_context: List[int] = field(
        default_factory=list,
        metadata={"help": "The encoding of the current conversation history used during planning."})
    agents_pos: List[int] = field(
        default_factory=lambda: [0, 0],
        metadata={"help": "The position of the two agents in the grid (row-major encoding)."})
    agents_items: List[int] = field(
        default_factory=lambda: [ITEM_NULL, ITEM_NULL],
        metadata={"help": "The items carried by the two agents, if any."})
    agents_visitation: Dict[int, List[int]] = field(
        default_factory=dict,
        metadata={"help": "List of grid cells visited by each agent thus far."})
    boxes_pos: List[int] = field(
        default_factory=lambda: [0, 0],
        metadata={"help": "The position of the two boxes in the grid (row-major encoding)."})
    boxes_content: List[List[int]] = field(
        default_factory=lambda: [[ITEM_GREEN, ITEM_YELLOW], [ITEM_PINK, ITEM_YELLOW]],
        metadata={"help": "The items in the two boxes, sorted by their value (descending)."})
    boxes_text: Dict[int, str] = field(
        default_factory=lambda: {ITEM_PINK: "PINK", ITEM_GREEN: "GREEN", ITEM_YELLOW: "YELLOW"},
        metadata={"help": "The text representation of the box contents."})
    dest_pos: List[int] = field(
        default_factory=lambda: [1, 2, 3],
        metadata={"help": "The position of three destinations (pink, green, yellow) in the grid (row-major encoding)."})
    dest_text: Dict[int, str] = field(
        default_factory=lambda: {LOC_DEST_PINK: "PINK", LOC_DEST_GREEN: "GREEN", LOC_DEST_YELLOW: "YELLOW"},
        metadata={"help": "The text representation of the destination positions."})
    reward_info: Dict[int, Dict] = field(
        default_factory=lambda: {agent_id: {} for agent_id in range(NUM_AGENTS)},
        metadata={"help": (
            "Realised rewards and their probabilities, per agent. Used during posterior sampling."
            "Note that this is very hacky and should be replaced with an 'info' dictionary akin "
            "to Gym interface that is collected within `sample_trajectory` method. However, this "
            "would require a significant refactoring of the environment interface, so we leave it "
            "as is for the time being."
        )}
    )

    @property
    def terminal(self) -> bool:
        # state is terminal in case an instruction is not valid (e.g., when a planner issues an invalid instruction)
        if self.mode in ["a1", "a2"] and not self.inst in list(Instruction):
            return True
        elif self.mode == "multi-agent-w-planner" and self.inst is not None and any(inst is None for inst in self.inst):
            return True

        if self.mode == "multi-agent-w-planner":
            # state is terminal if all agents have reached a destination
            return all(self.agents_pos[agent_id] in self.dest_pos for agent_id in range(NUM_AGENTS))
        else:
            # otherwise, state is terminal if the instruction is completed
            return self.instruction_completed

    @property
    def instruction_completed(self) -> bool:
        if self.mode == "multi-agent-w-planner":
            return all(self.is_instruction_completed(agent_id, shallow_comparison=True) for agent_id in range(NUM_AGENTS))
        elif self.mode == "a2":
            return self.is_instruction_completed(agent_id=1)
        else:
            return self.is_instruction_completed(agent_id=0)

    def to_obs(self, agent_id: int, batched: bool = False) -> Dict[str, Any]:
        # construct the observation dictionary
        inst = Instruction.from_str(self.inst[agent_id]) if self.mode == "multi-agent-w-planner" else Instruction.from_str(self.inst)
        obs = {
            "mask": np.full(len(GridWorldAction), True),
            "inst": inst.one_hot() if inst is not None else None,
            "pos": self.agents_pos[agent_id],
            "item": np.array([1 if i == self.agents_items[agent_id] else 0 for i in range(4)]),
        }

        # disable pickup actions unless the agent is explicitly instructed to pickup an item and is standing on a box
        pickup_inst = [Instruction.pickup_pink, Instruction.pickup_green, Instruction.pickup_yellow]
        pickup_act = [GridWorldAction.pickup_pink, GridWorldAction.pickup_green, GridWorldAction.pickup_yellow]
        if inst not in pickup_inst or self.agents_pos[agent_id] not in self.boxes_pos:
            for act in pickup_act:
                obs["mask"][act] = False

        # batch the items if requested
        if batched:
            obs = {key: None if item is None else np.array([item]) if isinstance(item, int) else item[None, :] for key, item in obs.items()}

        return obs

    def is_instruction_completed(self, agent_id: int, shallow_comparison: bool = False) -> bool:
        inst = self.inst[agent_id] if isinstance(self.inst, Tuple) else self.inst
        if inst in [Instruction.examine_box_1, Instruction.examine_box_2]:
            box_loc = LOC_BOX_1 if inst == Instruction.examine_box_1 else LOC_BOX_2
            return self.agents_pos[agent_id] == box_loc if not shallow_comparison else self.agents_pos[agent_id] in [LOC_BOX_1, LOC_BOX_2]
        elif inst in [Instruction.pickup_pink, Instruction.pickup_green, Instruction.pickup_yellow]:
            item = ITEM_PINK if inst == Instruction.pickup_pink else ITEM_GREEN if inst == Instruction.pickup_green else ITEM_YELLOW
            return self.agents_items[agent_id] == item if not shallow_comparison else self.agents_items[agent_id] != ITEM_NULL
        elif inst in [Instruction.goto_pink, Instruction.goto_green, Instruction.goto_yellow]:
            loc = self.dest_pos[0] if inst == Instruction.goto_pink else self.dest_pos[1] if inst == Instruction.goto_green else self.dest_pos[2]
            return self.agents_pos[agent_id] == loc if not shallow_comparison else self.agents_pos[agent_id] in self.dest_pos
        else:
            return False

    def __str__(self):
        return (
            f"State(inst={self.inst}, "
            f"agents_pos={self.agents_pos}, "
            f"agents_items={self.agents_items}, "
            f"boxes_pos={self.boxes_pos}, "
            f"boxes_content={self.boxes_content}, "
            f"dest_pos={self.dest_pos})")


class GridWorldTrajectory(Trajectory):
    def outcome(self) -> float:
        return self.total_reward(discount=0.99)

    def get_corridor_penalties(self, agent_id: int, corridor: Literal["pink", "green", "yellow"]) -> List[float]:
        return list(reversed([self.states[i].reward_info[agent_id]["reward"] - REWARDS["cell"]
                for i in range(self.horizon) if self.states[i].agents_pos[agent_id] in LOC_CORRIDORS[corridor]]))

    def render(self) -> str:
        result = f"GridWorld Trajectory {self.id}: "

        # show initial box content
        labels = {ITEM_NULL: "NULL", ITEM_YELLOW: "YELLOW", ITEM_GREEN: "GREEN", ITEM_PINK: "PINK"}
        result += f"Box 1 {labels[self.states[0].boxes_content[0][0]], labels[self.states[0].boxes_content[0][1]]}; "
        result += f"Box 2 {labels[self.states[0].boxes_content[1][0]], labels[self.states[0].boxes_content[1][1]]};\n"

        # show planner's and agent's actions and rewards
        for i, (state, action, rew) in enumerate(zip(self.states, self.actions, self.rewards)):
            if state.terminal:
                assert all(a is None for a in action)
                result += f"    Step: {i}; Goal Reward: {rew}; Total Reward: {self.total_reward(discount=1.0)};\n"
            elif state.stage == "planning" and action[-1] is not None:
                inventory = f"{labels[state.agents_items[0]]}, {labels[state.agents_items[1]]}"
                result += f"    Step: {i}; Reporter: {action[-1].obs} Position (A1, A2): {state.agents_pos}; Inventory (A1, A2): {inventory};\n"
                result += f"             Planner: {action[-1].inst}; Reward {rew};\n"
            elif state.stage == "execution":
                a1_act = GridWorldAction(action[0]).name if action[0] is not None else None
                a2_act = GridWorldAction(action[1]).name if action[1] is not None else None
                reward = f"Reward: {rew} ("
                reward = f"{reward}A1: {state.reward_info[0]['reward']}" if 0 in state.reward_info and "reward" in state.reward_info[0] else reward
                reward = f"{reward}, A2: {state.reward_info[1]['reward']}" if 1 in state.reward_info and "reward" in state.reward_info[1] else reward
                reward = f"{reward})".replace("()", "")
                result += f"    Step: {i}; Actors (A1, A2): {a1_act}, {a2_act}; Position (A1, A2): {state.agents_pos}; {reward}\n"

        return result


class GridWorld(CausalEnv):
    def __init__(
        self,
        mode: Literal["a1", "a2", "multi-agent-w-planner"] = "multi-agent-w-planner",
        init_mode: Optional[Literal["analysis"]] = None,
        horizon: int = 40,
        act_noise_model: NoiseModel = NullNoiseModel(),
        env_noise_model: NoiseModel = NullNoiseModel(),
    ):
        super().__init__(act_noise_model=act_noise_model, env_noise_model=env_noise_model)
        self.mode = mode
        self.horizon = horizon
        self.init_mode = init_mode
        self.walls = LOC_WALLS
        self.corridors = LOC_CORRIDORS
        self.rewards = REWARDS
        self.num_agents = 1 if mode == "a1" else 2 if mode == "a2" else 3

    def reset(self, rng: Optional[Generator] = None, inst: Optional[Union[Instruction, Tuple[Instruction]]] = None, env_noise: Optional[Dict[str, List[NoiseModel]]] = None):
        assert inst is not None if self.mode != "multi-agent-w-planner" else True, "A1 and A2 modes expect an instruction."

        # initialize agents and contents of the two boxes
        rng = rng if rng is not None else np.random.default_rng()
        a1_pos, a2_pos = self._get_initial_agents_pos(rng, inst)
        box_1, box_2 = self._get_initial_boxes_content(rng, inst)
        a1_item, a2_item = self._get_initial_agents_item(rng, inst)

        # construct the initial state
        return State(
            mode=self.mode,
            stage="planning" if self.mode == "multi-agent-w-planner" else "execution",
            inst=inst.value if isinstance(inst, Instruction) else (inst[0].value, inst[1].value) if isinstance(inst, Tuple) else None,
            agents_pos=[a1_pos, a2_pos],
            agents_visitation={agent_id: [agent_pos] for agent_id, agent_pos in enumerate([a1_pos, a2_pos])},
            boxes_pos=[LOC_BOX_1, LOC_BOX_2],
            dest_pos=[LOC_DEST_PINK, LOC_DEST_GREEN, LOC_DEST_YELLOW],
            boxes_content=[box_1, box_2],
            agents_items=[a1_item, a2_item],
        )

    def step(self, state: State, actions: Tuple[Union[int, PlannerAction], ...], env_noise: Dict[str, List[NoiseModel]], rng: Optional[Generator] = None):
        if self.mode == "multi-agent-w-planner":
            return self._step_with_planner(state, actions, env_noise, rng)
        elif self.mode == "a1" or self.mode == "a2":
            return self._step_single_agent(state, actions, env_noise, rng)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

    def sample_trajectory(
        self,
        agents: List[GridWorldActor],
        act_noise: Optional[List[List[NoiseModel]]] = None,
        env_noise: Optional[List[Dict[str, List[NoiseModel]]]] = None,
        initial_state: Dict[str, Any] = None,
        do_operators: Optional[List[Dict[int, int]]] = None,
        rng: Optional[Generator] = None,
        horizon: Optional[int] = None,
        pad: bool = False,
    ) -> GridWorldTrajectory:
        t = super().sample_trajectory(agents, act_noise, env_noise, initial_state, do_operators, rng, horizon, pad)
        t = GridWorldTrajectory(id=t.id, states=t.states, actions=t.actions, rewards=t.rewards, probs=t.probs, env_noise=t.env_noise, act_noise=t.act_noise)
        return t

    def _get_acting_agents(self, state: State, agents: List[CausalActor]) -> List[CausalActor]:
        if state.terminal:
            # no agent can act in a terminal state
            return []
        if state.stage == "planning":
            # we assume a planner is always the last agent in the list
            return [agents[-1]]
        else:
            # in the execution stage, an agent can act if its instruction is not completed
            return [a for a in agents[:2] if not state.is_instruction_completed(a.id)]

    def _step_single_agent(self, state: State, actions: Tuple[Union[int, PlannerAction], ...], env_noise: Dict[str, List[NoiseModel]], rng: Optional[Generator] = None):
        assert len(actions) == 1, "A1/A2 mode expects a single action."
        assert isinstance(actions[0], int), "A1/A2 mode expects an integer action."
        assert isinstance(state.inst, str) and state.inst != "", "A1/A2 mode expects a single agent instruction (see the `reset` method)."
        next_state, action, agent_id = deepcopy(state), actions[0], 0 if state.mode == "a1" else 1

        # handle movement and pickup actions
        if state.agents_pos[agent_id] not in state.dest_pos:
            next_state = self._transition_agent_pos(next_state, agent_id=agent_id, action=action)
            next_state = self._transition_agent_inventory(next_state, agent_id=agent_id, action=action)

        # calculate the reward
        reward, info = self._get_reward(next_state, env_noise, rng)
        next_state.reward_info = info

        return next_state, reward

    def _step_with_planner(self, state: State, actions: Tuple[Union[int, PlannerAction], ...], env_noise: Dict[str, List[NoiseModel]], rng: Optional[Generator] = None):
        assert not state.terminal, "Cannot step in a terminal state"
        next_state = deepcopy(state)

        if state.stage == "planning":
            # if in the planning stage, update the state with planner's instruction
            assert actions[0] is None and actions[1] is None, "Agents do not act during planning"

            next_state.inst_context = actions[-1].inst_context
            next_state.inst = Instruction.from_str(actions[-1].inst[0]), Instruction.from_str(actions[-1].inst[1])
            next_state.stage = "execution"

            reward, info = self._get_reward(next_state, env_noise, rng)
            next_state.reward_info = info

        if state.stage == "execution":
            assert actions[-1] is None, "Planner does not act during execution"

            # handle movement actions for both agents
            for agent_id in range(NUM_AGENTS):
                if state.agents_pos[agent_id] not in state.dest_pos and actions[agent_id] is not None:
                    next_state = self._transition_agent_pos(next_state, agent_id=agent_id, action=actions[agent_id])
                    next_state = self._transition_agent_inventory(next_state, agent_id=agent_id, action=actions[agent_id])

            # move to the planning stage, if the agents have completed their instruction
            if next_state.instruction_completed and not next_state.terminal:
                next_state.stage = "planning"
                return next_state, 0.0

            # calculate reward
            reward, info = self._get_reward(next_state, env_noise, rng)
            next_state.reward_info = info

        return next_state, reward

    def _get_reward(self, state: State, env_noise: Dict[str, List[NoiseModel]] = {}, rng: Optional[Generator] = None) -> Tuple[float, Dict]:
        if state.terminal:
            if state.mode == "multi-agent-w-planner":
                reward = 0.0
                for agent_id in range(NUM_AGENTS):
                    if state.agents_pos[agent_id] == LOC_DEST_PINK and state.agents_items[agent_id] == ITEM_PINK:
                        reward += self.rewards["corridors"]["pink"]["goal"]
                    elif state.agents_pos[agent_id] == LOC_DEST_GREEN and state.agents_items[agent_id] == ITEM_GREEN:
                        reward += self.rewards["corridors"]["green"]["goal"]
                    elif state.agents_pos[agent_id] == LOC_DEST_YELLOW and state.agents_items[agent_id] == ITEM_YELLOW:
                        reward += self.rewards["corridors"]["yellow"]["goal"]
                    elif state.inst[agent_id] in [Instruction.examine_box_1, Instruction.examine_box_2, Instruction.pickup_pink, Instruction.pickup_green, Instruction.pickup_yellow]:
                        reward += self.rewards["cell"]
                return reward, {}
            else:
                if state.inst == Instruction.goto_pink:
                    return self.rewards["corridors"]["pink"]["goal"], {}
                elif state.inst == Instruction.goto_green:
                    return self.rewards["corridors"]["green"]["goal"], {}
                elif state.inst == Instruction.goto_yellow:
                    return self.rewards["corridors"]["yellow"]["goal"], {}
                else:
                    goal_mean = sum([self.rewards["corridors"][corridor]["goal"] for corridor in ["pink", "green", "yellow"]]) / 3
                    return goal_mean, {}

        # if we do have corridor penalties, incorporate them based on the selected phase
        total_reward = 0.0
        info = {agent_id: {} for agent_id in range(NUM_AGENTS)}

        for agent_id in range(NUM_AGENTS):
            # when training first or the second agent, only consider its reward
            if state.mode == "a1" and agent_id == 1:
                continue
            elif state.mode == "a2" and agent_id == 0:
                continue

            # if a player is not standing on a corridor cell, apply the empty cell reward
            if not any(state.agents_pos[agent_id] in cells for cells in self.corridors.values()):
                info[agent_id] = {"reward": self.rewards["cell"]}
                total_reward += info[agent_id]["reward"]
                continue

            # otherwise, first apply a fixed cell penalty
            reward = self.rewards["cell"]

            # then apply a stochastic corridor penalty
            corridor = "pink" if state.agents_pos[agent_id] in self.corridors["pink"] else "green" if state.agents_pos[agent_id] in self.corridors["green"] else "yellow"
            matching_item = ITEM_PINK if corridor == "pink" else ITEM_GREEN if corridor == "green" else ITEM_YELLOW
            rewards_key = "rewards_reduced" if state.agents_items[agent_id] == matching_item else "rewards"

            rewards_key = f"{rewards_key}_c{self.corridors[corridor].index(state.agents_pos[agent_id])}" if corridor == "green" else rewards_key
            probs = list(map(lambda i: i[1], self.rewards["corridors"][corridor][rewards_key]))
            rewards = list(map(lambda i: i[0], self.rewards["corridors"][corridor][rewards_key]))

            if "reward" in env_noise:
                reward += rewards[env_noise["reward"][agent_id].choice(probs, rng=rng)]
            else:
                reward += rng.choice(rewards, p=probs).item()

            info[agent_id] = {"reward": reward, "probs": probs, "rewards": rewards}
            total_reward += reward

        return total_reward, info

    def _get_initial_agents_pos(self, rng: Generator, inst: Optional[Union[Instruction, Tuple[Instruction]]] = None) -> Tuple[int]:
        box_1_row, box_1_col = LOC_BOX_1 // GRID_COLS, LOC_BOX_1 % GRID_COLS
        box_2_row, box_2_col = LOC_BOX_2 // GRID_COLS, LOC_BOX_2 % GRID_COLS

        agent_pos_box_1 = (box_1_row + 1) * GRID_COLS + box_1_col # spawns the agent below the first box
        agent_pos_box_2 = (box_2_row - 1) * GRID_COLS + box_2_col # spawns the agent above the second box

        if self.mode in ["a1", "a2"] and inst in [Instruction.examine_box_1, Instruction.examine_box_2]:
            # when training A1/A2, spawn the agent next to the box it's supposed to examine
            agent_pos = agent_pos_box_1 if inst == Instruction.examine_box_1 else agent_pos_box_2
            return (agent_pos, -1) if self.mode == "a1" else (-1, agent_pos)
        elif self.mode == "multi-agent-w-planner":
            # in the multi-agent with planner mode, the agents are always initialized next to their respective boxes
            return agent_pos_box_1, agent_pos_box_2
        else:
            # for goto and pickup instructions, we fix the locations of the two agents
            return (LOC_BOX_1, LOC_BOX_2)

    def _get_initial_agents_item(self, rng: Generator, inst: Optional[Union[Instruction, Tuple[Instruction]]] = None) -> List[int]:
        if self.mode in ["a1", "a2"] and inst in [Instruction.goto_pink, Instruction.goto_green, Instruction.goto_yellow]:
            # for goto instructions, ensure an agent is carrying an item
            item = rng.choice([ITEM_PINK, ITEM_GREEN, ITEM_YELLOW], size=1).item()
            return (item, ITEM_NULL) if self.mode == "a1" else (ITEM_NULL, item)
        else:
            # in all other cases, the agents are initialized without any items
            return ITEM_NULL, ITEM_NULL

    def _get_initial_boxes_content(self, rng: Generator, inst: Optional[Instruction] = None) -> Tuple[List[int], List[int]]:
        if isinstance(inst, Instruction) and inst in [Instruction.pickup_pink, Instruction.pickup_green, Instruction.pickup_yellow]:
            # in A1/A2 mode, when requesting an agent to pickup an item, ensure the box contains the requested item
            requested_item = ITEM_PINK if inst == Instruction.pickup_pink else ITEM_GREEN if inst == Instruction.pickup_green else ITEM_YELLOW
            box_1 = sorted([requested_item, rng.choice([ITEM_PINK, ITEM_GREEN, ITEM_YELLOW], size=1).item()], reverse=True)
            box_2 = sorted([requested_item, rng.choice([ITEM_PINK, ITEM_GREEN, ITEM_YELLOW], size=1).item()], reverse=True)
            return box_1, box_2
        elif self.init_mode == "analysis":
            # for our analysis, we assume that the first agent has a pink option, and the second agent has pink and green options
            box_1 = [ITEM_PINK, rng.choice([ITEM_PINK, ITEM_GREEN, ITEM_YELLOW], size=1).item()]
            box_2 = [ITEM_PINK, ITEM_GREEN]
            return box_1, box_2
        else:
            box_1 = sorted(rng.choice([ITEM_PINK, ITEM_GREEN, ITEM_YELLOW], size=2).tolist(), reverse=True)
            box_2 = sorted(rng.choice([ITEM_PINK, ITEM_GREEN, ITEM_YELLOW], size=2).tolist(), reverse=True)
            return box_1, box_2

    def _transition_agent_pos(self, state: State, agent_id: int, action: GridWorldAction) -> State:
        # for non-moving actions, do not make any changes
        if action in [GridWorldAction.pickup_pink, GridWorldAction.pickup_green, GridWorldAction.pickup_yellow, GridWorldAction.noop]:
            return state

        # calculate the new agent position
        ag_row, ag_col = state.agents_pos[agent_id] // GRID_COLS, state.agents_pos[agent_id] % GRID_COLS
        ag_row_offset, ag_col_offset = GridWorldAction.offset(action)
        ag_row_new, ag_col_new = ag_row + ag_row_offset, ag_col + ag_col_offset

        # prevent the agent from moving into a wall
        if ag_row_new * GRID_COLS + ag_col_new in self.walls:
            ag_row_new, ag_col_new = ag_row, ag_col

        # prevent the agent from moving out of the grid world
        if ag_row_new < 0 or ag_row_new >= GRID_ROWS:
            ag_row_new = ag_row
        if ag_col_new < 0 or ag_col_new >= GRID_COLS:
            ag_col_new = ag_col

        # prevent the agent from moving into the other agent, outside of the final destinations
        if ag_row_new * GRID_COLS + ag_col_new == state.agents_pos[1 - agent_id] and state.agents_pos[1 - agent_id] not in state.dest_pos:
            ag_row_new = ag_row
            ag_col_new = ag_col

        # update the agent's position and return the new state
        state.agents_pos[agent_id] = ag_row_new * GRID_COLS + ag_col_new
        state.agents_visitation[agent_id].append(state.agents_pos[agent_id])
        return state

    def _transition_agent_inventory(self, state: State, agent_id: int, action: GridWorldAction) -> State:
        if action not in [GridWorldAction.pickup_pink, GridWorldAction.pickup_green, GridWorldAction.pickup_yellow]:
            # nothing to do in case an agent does not try to pickup an item
            return state

        if state.agents_pos[agent_id] not in state.boxes_pos:
            # agent cannot pickup an item unless it's standing on a box
            return state

        box = state.boxes_pos.index(state.agents_pos[agent_id])
        requested_item = ITEM_PINK if action == GridWorldAction.pickup_pink else ITEM_GREEN if action == GridWorldAction.pickup_green else ITEM_YELLOW

        if requested_item not in state.boxes_content[box]:
            # agent cannot pickup an item unless it is available in the box
            return state

        if state.agents_items[agent_id] == ITEM_NULL:
            # if the agent is not carrying an item, pick it up from the box
            state.agents_items[agent_id] = requested_item
            state.boxes_content[box][state.boxes_content[box].index(requested_item)] = ITEM_NULL
            state.boxes_content[box] = sorted(state.boxes_content[box], reverse=True)
        else:
            # if the agent is already holding an item, replace the item in the box with the item in the agent's inventory
            box_item_index = state.boxes_content[box].index(requested_item)
            box_item = state.boxes_content[box][box_item_index]
            state.boxes_content[box][box_item_index] = state.agents_items[agent_id]
            state.boxes_content[box] = sorted(state.boxes_content[box], reverse=True)
            state.agents_items[agent_id] = box_item

        return state

    def _get_act_noise(self, agents: List[CausalActor], state: Optional[Dict[str, Any]] = None, trajectory: Optional[Trajectory] = None, time_step: Optional[int] = None, rng: Optional[Generator] = None) -> List[NoiseModel]:
        # agent's policies are deterministic in this environment
        return [self.act_noise_model.sample() for _ in range(self.num_agents)]

    def _get_env_noise(self, trajectory: Optional[Trajectory] = None, time_step: Optional[int] = None, rng: Optional[Generator] = None) -> Dict[str, NoiseModel]:
        assert (not trajectory and time_step is None) or (trajectory and time_step is not None), "Either both trajectory and time_step should be passed (posterior sampling), or none of them (regular sampling)"

        # === perform regular noise sampling, when no trajectory is passed ===
        if not trajectory:
            return {"reward": {agent_id: self.env_noise_model.sample(rng=rng) for agent_id in range(NUM_AGENTS)}}

        # === perform posterior sampling, when a trajectory is given ===
        result = {"reward": {agent_id: self.env_noise_model.sample(rng=rng) for agent_id in range(NUM_AGENTS)}}

        if self.mode != "multi-agent-w-planner":
            # no stochastic rewards are applied in the A1/A2 mode
            return result

        state = trajectory.states[time_step]

        if not any(state.agents_pos[agent_id] in cells for cells in self.corridors.values() for agent_id in range(NUM_AGENTS)):
            # no stochastic rewards are applied when the agents are not standing on a corridor cell
            return result

        for agent_id in range(NUM_AGENTS):
            if self.mode == "a2" and agent_id == 0:
                continue
            if not any(state.agents_pos[agent_id] in cells for cells in self.corridors.values()):
                continue
            if agent_id not in state.reward_info or "probs" not in state.reward_info[agent_id]:
                continue

            probs = state.reward_info[agent_id]["probs"]
            rewards = state.reward_info[agent_id]["rewards"]

            realised = state.reward_info[agent_id]["reward"] - self.rewards["cell"]
            realised = rewards.index(realised)

            result["reward"][agent_id] = self.env_noise_model.sample_posterior(probs, realised, rng=rng)

            sampled_reward = rewards[result["reward"][agent_id].choice(probs)]
            assert sampled_reward == rewards[realised], f"Posterior sampling failed: expected agent's reward {realised} but got {sampled_reward}"

        calculated_reward = self._get_reward(state, result)[0]
        assert calculated_reward == trajectory.rewards[time_step], f"Posterior sampling failed: expected reward {trajectory.rewards[time_step]} but got {calculated_reward} at time-step {time_step}"
        return result


class GridWorldGym(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        env: GridWorld,
        instruction: Optional[Union[Instruction, Tuple[Instruction]]] = None,
        weighted_sampling: bool = False,
        render_mode: Optional[Literal["human", "rgb_array"]] = None,
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # initialize base environment properties
        self.env = env
        self.render_mode = render_mode
        self.instruction = instruction
        self.weighted_sampling = weighted_sampling
        self.action_space = gym.spaces.Discrete(len(GridWorldAction), seed=self._np_random or np.random.default_rng())
        self.observation_space = gym.spaces.Dict({
            "inst": gym.spaces.Box(low=0, high=1, shape=(len(Instruction),), dtype=np.float32),
            "pos": gym.spaces.Discrete(GRID_ROWS * GRID_COLS),
            "item": gym.spaces.Discrete(4),
        })

        # initialize render properties
        self.window = None # reference to the PyGame window
        self.clock = None # clock used for environment rendering at a correct framerate
        self.sprites = {} # dictionary containing all the initialized sprites to be rendered
        self.window_size = 32 * GRID_COLS, 32 * GRID_ROWS
        self.cell_size = self.window_size[0] // GRID_COLS, self.window_size[1] // GRID_ROWS
        self.assets = {
            "cell": Path(__file__).parent / ".." / ".." / "assets" / "grid" / "sprites" / "cell.png",
            "wall": Path(__file__).parent / ".." / ".." / "assets" / "grid" / "sprites" / "wall.png",
            "box_0": Path(__file__).parent / ".." / ".." / "assets" / "grid" / "sprites" / "box_0.png",
            "box_1": Path(__file__).parent / ".." / ".." / "assets" / "grid" / "sprites" / "box_1.png",
            "agent_0": Path(__file__).parent / ".." / ".." / "assets" / "grid" / "sprites" / "agent_0.png",
            "agent_1": Path(__file__).parent / ".." / ".." / "assets" / "grid" / "sprites" / "agent_1.png",
            "dest_pink": Path(__file__).parent / ".." / ".." / "assets" / "grid" / "sprites" / "dest_pink.png",
            "dest_green": Path(__file__).parent / ".." / ".." / "assets" / "grid" / "sprites" / "dest_green.png",
            "dest_yellow": Path(__file__).parent / ".." / ".." / "assets" / "grid" / "sprites" / "dest_yellow.png",
            "corridor_pink": Path(__file__).parent / ".." / ".." / "assets" / "grid" / "sprites" / "corridor_pink.png",
            "corridor_green": Path(__file__).parent / ".." / ".." / "assets" / "grid" / "sprites" / "corridor_green.png",
            "corridor_yellow": Path(__file__).parent / ".." / ".." / "assets" / "grid" / "sprites" / "corridor_yellow.png",
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed, options=options)

        if self.env.mode == "multi-agent-w-planner":
            # when using a planner, we only use the causal environment from above
            raise NotImplementedError(f"Mode {self.env.mode} cannot be used with a Gym environment.")

        # for training agents, in all scenarios, we sample an instruction
        rng = self._np_random or np.random.default_rng(seed)
        weights = [Instruction.get_sampling_weight(i) for i in Instruction] if self.weighted_sampling else None
        inst = list(Instruction)[rng.choice(range(len(list(Instruction))), p=weights)]

        # if a custom  instruction is passed, use it instead
        if self.instruction or options and "inst" in options:
            inst = options["inst"] if options and "inst" in options else self.instruction

        # construct the observation of the agent we are training
        agent_id = 0 if self.env.mode == "a1" else 1
        self.state = self.env.reset(rng=rng, inst=inst)
        obs, info = self.state.to_obs(agent_id=agent_id), {}

        return obs, info

    def step(self, action: Union[int, np.ndarray, Tuple[int]]):
        rng = self._np_random or np.random.default_rng()

        action = action if isinstance(action, int) else action.item()
        actions = (action, )

        next_state, reward = self.env.step(self.state, actions=actions, env_noise={}, rng=rng)

        agent_id = 0 if self.env.mode == "a1" else 1
        terminated, truncated = next_state.terminal, False
        obs, info = next_state.to_obs(agent_id=agent_id), {}

        self.state = next_state
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array" or self.render_mode == "human":
            return self._render_frame(self.render_mode)

    def close(self):
        self.window = None
        self.clock = None
        self.sprites = {}
        pygame.display.quit()
        pygame.quit()

    def keypress(self) -> GridWorldAction:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return GridWorldAction.up
                elif event.key == pygame.K_DOWN:
                    return GridWorldAction.down
                elif event.key == pygame.K_LEFT:
                    return GridWorldAction.left
                elif event.key == pygame.K_RIGHT:
                    return GridWorldAction.right
                elif event.key == pygame.K_1:
                    return GridWorldAction.pickup_pink
                elif event.key == pygame.K_2:
                    return GridWorldAction.pickup_green
                elif event.key == pygame.K_3:
                    return GridWorldAction.pickup_yellow
        return GridWorldAction.noop

    def _render_frame(self, mode: Literal["rgb_array", "human"]):
        if self.window is None:
            # initialize general PyGame modules
            pygame.init()

            if mode == "human":
                # initialize the PyGame display
                pygame.display.init()
                pygame.display.set_caption("Grid World")
                self.window = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                # when rendering to an array, skip the display initialization
                self.window = pygame.Surface(self.window_size)

        if self.clock is None:
            # initialize PyGame clock for capping the framerate
            self.clock = pygame.time.Clock()

        # load and initialize sprites
        self.sprites = {
            key: pygame.transform.scale(pygame.image.load(file_name), self.cell_size) for key, file_name in self.assets.items()}

        # render the grid world
        for y in range(GRID_ROWS):
            for x in range(GRID_COLS):
                grid_pos = y * GRID_COLS + x

                # select the appropriate sprite to render
                if grid_pos in self.state.agents_pos:
                    agent_id = self.state.agents_pos.index(grid_pos)
                    sprite = self.sprites[f"agent_{agent_id}"]
                elif grid_pos in self.state.boxes_pos:
                    box_id = self.state.boxes_pos.index(grid_pos)
                    sprite = self.sprites[f"box_{box_id}"]
                elif grid_pos in self.state.dest_pos:
                    dest_name = "pink" if self.state.dest_pos[0] == grid_pos else "green" if self.state.dest_pos[1] == grid_pos else "yellow"
                    sprite = self.sprites[f"dest_{dest_name}"]
                elif grid_pos in self.env.walls:
                    sprite = self.sprites["wall"]
                elif any(grid_pos in cells for cells in self.env.corridors.values()):
                    corridor = "pink" if grid_pos in self.env.corridors["pink"] else "green" if grid_pos in self.env.corridors["green"] else "yellow"
                    sprite = self.sprites[f"corridor_{corridor}"]
                else:
                    sprite = self.sprites["cell"]

                # render the sprite
                top_left_x, top_left_y = x * self.cell_size[0], y * self.cell_size[1]
                self.window.blit(sprite, (top_left_x, top_left_y))

        # update the display or return the rendered pixels array
        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.keypress()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            pixels = np.array(pygame.surfarray.pixels3d(self.window))
            return np.transpose(pixels, axes=(1, 0, 2))
