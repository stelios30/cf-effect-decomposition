import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tianshou.data import Batch
from tianshou.policy import DQNPolicy, PPOPolicy
from tianshou.utils.net.common import Net

from ced.tools.llm import LLMGateway, LLMResponse
from ced.tools.noise import NoiseModel

from .causal import CausalActor

if TYPE_CHECKING:
    from ced.envs.grid import State


@dataclass
class PlannerAction:
    inst: Tuple[str] = field(
        metadata={"help": "Parsed instructions for each agent."})
    obs: str = field(
        default="",
        metadata={"help": "The observation (prompt) used to generate this action."})
    inst_raw: str = field(
        default="",
        metadata={"help": "Raw instruction as issued by the planner."})
    inst_context: List[int] = field(
        default_factory=list,
        metadata={"help": "The encoding of the current conversation history used during planning."})


class GridWorldAction(int, Enum):
    up = 0
    down = 1
    left = 2
    right = 3
    pickup_pink = 4
    pickup_green = 5
    pickup_yellow = 6
    noop = 7

    @classmethod
    def offset(cls, action: "GridWorldAction") -> Tuple[int, int]:
        if action == cls.up:
            return -1, 0
        elif action == cls.down:
            return 1, 0
        elif action == cls.right:
            return 0, 1
        elif action == cls.left:
            return 0, -1

    def __str__(self) -> str:
        return self.name


class Reporter:
    def report(self, state: "State") -> str:
        # implements the reporter logic, which in our case is a deterministic function of the current state
        obs = []

        # report the state of the two agents
        for agent_id in range(2):
            if state.agents_pos[agent_id] == state.boxes_pos[0] and state.agents_items[agent_id] == 0:
                # if agent reached the first box and has no item, show the box content
                obs.append(f"A{agent_id+1} ({state.boxes_text[state.boxes_content[0][0]]} {state.boxes_text[state.boxes_content[0][1]]});")
            elif state.agents_pos[agent_id] == state.boxes_pos[0] and state.agents_items[agent_id] != 0:
                # if agent reached the first box and has an item, show the item
                obs.append(f"A{agent_id+1} has {state.boxes_text[state.agents_items[agent_id]]};")
            elif state.agents_pos[agent_id] == state.boxes_pos[1] and state.agents_items[agent_id] == 0:
                # if agent reached the second box and has no item, show the box content
                obs.append(f"A{agent_id+1} ({state.boxes_text[state.boxes_content[1][0]]} {state.boxes_text[state.boxes_content[1][1]]});")
            elif state.agents_pos[agent_id] == state.boxes_pos[1] and state.agents_items[agent_id] != 0:
                # if agent reached the second box and has an item, show the item
                obs.append(f"A{agent_id+1} has {state.boxes_text[state.agents_items[agent_id]]};")
            elif state.agents_pos[agent_id] in state.dest_pos:
                # if agent is at a destination, show its location
                obs.append(f"A{agent_id+1} at {state.dest_text[state.agents_pos[agent_id]]};")
            else:
                # otherwise indicate that the agent has respawned
                obs.append(f"A{agent_id+1} respawn;")

        # construct the final observation
        return f"{' '.join(obs)}"


class PlannerActor(CausalActor):
    def __init__(self, id: int, gateway: LLMGateway, reporter: Reporter):
        super().__init__(id)
        self.gateway = gateway
        self.reporter = reporter

    def policy(self, state: "State") -> List[float]:
        return [1.0] # LLM actor has dummy probabilities

    def action(self, state: "State", act_noise: NoiseModel, return_probs: bool = False) -> Union[PlannerAction, Tuple[PlannerAction, List[float]]]:
        obs = self.reporter.report(state)
        response = self.plan(obs, state.inst_context)
        inst_raw = response.response

        # parse the instruction for each agent
        inst_str = inst_raw.lower().replace("inst:", "").strip(" ;\n")
        inst_str = re.sub(r"a\d+", "", inst_str)
        inst = inst_str.split(";")
        inst = tuple([i.strip() for i in inst])

        action = PlannerAction(inst=inst, inst_raw=inst_raw, inst_context=response.context, obs=obs)
        probs = self.policy(state)
        return action if not return_probs else (action, probs)

    def plan(self, obs: str, context: List[int] = []) -> LLMResponse:
        return self.gateway.generate(prompt=obs, context=context)


class HardcodedPlannerActor(CausalActor):
    def policy(self, state: "State") -> List[float]:
        return [1.0] # LLM actor has dummy probabilities

    def action(self, state: "State", act_noise: NoiseModel, return_probs: bool = False) -> Union[PlannerAction, Tuple[PlannerAction, List[float]]]:
        def _get_instruction(agent_id: int) -> str:
            if state.agents_pos[agent_id] in state.boxes_pos and state.agents_items[agent_id] == 0:
                # if the agent is standing on the box and does not have an item, issue pickup instruction for the most valuable item
                box_id = state.boxes_pos.index(state.agents_pos[agent_id])
                item_label = state.boxes_text[max(state.boxes_content[box_id])].lower()
                return f"pickup {item_label}"
            elif state.agents_pos[agent_id] in state.boxes_pos and state.agents_items[agent_id] != 0:
                # if the agent is standing on the box and has an item, issue the goto instruction to the destination of the item its holding
                box_id = state.boxes_pos.index(state.agents_pos[agent_id])
                item_label = state.boxes_text[state.agents_items[agent_id]].lower()
                return f"goto {item_label}"
            else:
                # otherwise, issue an examine instruction
                return f"examine box {agent_id + 1}"

        action = PlannerAction(inst=(_get_instruction(agent_id=0), _get_instruction(agent_id=1)))
        return action if not return_probs else (action, self.policy(state))


class DQNGridWorldActorNet(Net):
    def __init__(self, *args, num_pos_embeddings: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_pos_embeddings = num_pos_embeddings

    def forward(self, obs, state = None, info = {}):
        agent_inst = torch.tensor(obs.inst, device=self.device)
        agent_pos = F.one_hot(torch.tensor(obs.pos, device=self.device), num_classes=self.num_pos_embeddings)
        agent_item = torch.tensor(obs.item, device=self.device)
        agent_obs = (agent_inst, agent_pos, agent_item)
        return super().forward(torch.cat(agent_obs, dim=1), state, info)


class DQNGridWorldPolicy(DQNPolicy):
    def __init__(self, *args, tau: Optional[float] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tau = tau # target network update rate

    def sync_weight(self) -> None:
        if self.tau is None:
            # copies the model weights to the target network
            self.model_old.load_state_dict(self.model.state_dict())
        else:
            # performs Polyak averaging, if requested
            for param, target_param in zip(self.model.parameters(), self.model_old.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class GridWorldActor(CausalActor):
    def __init__(self, id: int, policy: Union[DQNGridWorldPolicy, PPOPolicy]):
        super().__init__(id)
        self.policy_fn = policy

    def policy(self, state: "State") -> List[float]:
        batch = Batch(obs=state.to_obs(agent_id=self.id, batched=True), info={})
        act = self.policy_fn(batch=batch).act.item()

        probs = np.zeros((len(GridWorldAction), ))
        probs[act] = 1.0
        return probs

    def action(self, state: "State", act_noise: Optional[NoiseModel] = None, return_probs: bool = False) -> Union[int, Tuple[int, List[float]]]:
        probs = self.policy(state)
        action = np.argmax(probs).item()
        return action if not return_probs else (action, probs)


def make_dqn_policy(
    lr: float = 5e-4,
    tau: float = 1e-3,
    hidden_size: int = 64,
    hidden_depth: int = 2,
    num_pos_embeddings: int = 60,
    num_item_embeddings: int = 4,
    discount: float = 0.99,
    n_estimation_step: int = 5,
    target_update_freq: int = 100,
    device: str = "cuda",
):
    action_dim = len(GridWorldAction)
    state_dim = 8 + num_pos_embeddings + num_item_embeddings # one-hot encoded instruction, position and item

    net = DQNGridWorldActorNet(
        state_shape=state_dim, action_shape=action_dim,
        hidden_sizes=[hidden_size] * hidden_depth,
        num_pos_embeddings=num_pos_embeddings,
        device=device, softmax=False, activation=nn.LeakyReLU,
    )
    optim = torch.optim.Adam(net.parameters(), lr=lr, eps=15e-5)

    policy = DQNGridWorldPolicy(
        model=net,
        optim=optim,
        discount_factor=discount,
        estimation_step=n_estimation_step,
        tau=tau if target_update_freq == 1 else None,
        is_double=True,
        clip_loss_grad=False,
        target_update_freq=target_update_freq,
    )
    return policy.to(device)
