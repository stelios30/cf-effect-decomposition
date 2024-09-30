import torch
import pytest

from ced.envs.grid import GridWorld, GridWorldGym, ITEM_GREEN, ITEM_PINK, ITEM_YELLOW, LOC_DEST_GREEN, LOC_DEST_PINK, LOC_DEST_YELLOW, State, Instruction, LOC_BOX_1, LOC_BOX_2
from ced.actors.grid import GridWorldActor, PlannerAction
from ced.tools.utils import to_device


@pytest.fixture
def make_gw_env():
    def _factory(**kwargs):
        params = {"mode": "a1", "phase": None, "horizon": 25, **kwargs}
        return GridWorld(**params)
    return _factory


@pytest.fixture
def make_gw_gym_env(make_gw_env):
    def _factory(init_state=None, **kwargs):
        env = make_gw_env(**{"mode": "a1", **kwargs})

        teammate_policy = torch.load("./results/grid/a1_policy.pt", map_location="cpu")["policy"].eval()
        teammate_policy = to_device(teammate_policy, device="cpu")
        teammate_agent = GridWorldActor(id=0, policy=teammate_policy) if env.mode != "a1" else None

        env = GridWorldGym(env=env, teammate=teammate_agent)
        env.state = init_state
        return env
    return _factory


@pytest.fixture
def make_gw_state():
    def _factory(**kwargs):
        mode = kwargs.get("mode", "a1")
        return State(**{
            "mode": mode,
            "inst": Instruction.goto_pink if mode == "a1" else (Instruction.goto_pink.value, Instruction.goto_green.value),
            "agents_pos": [LOC_BOX_1, LOC_BOX_2],
            "agents_visitation": {0: [LOC_BOX_1], 1: [LOC_BOX_2]},
            "boxes_pos": [LOC_BOX_1, LOC_BOX_2],
            "dest_pos": [LOC_DEST_PINK, LOC_DEST_GREEN, LOC_DEST_YELLOW],
            "boxes_content": [[ITEM_PINK, ITEM_YELLOW], [ITEM_GREEN, ITEM_YELLOW]],
            **kwargs,
        })
    return _factory


@pytest.fixture
def make_gw_planner_act():
    def _factory(**kwargs):
        return PlannerAction(**{
            "inst": ("goto pink", "goto yellow"),
            "inst_raw": "inst: agent 1 goto PINK; agent 2 goto YELLOW;",
            "inst_context": [5, 9, 1, 3, 5],
            **kwargs,
        })
    return _factory
