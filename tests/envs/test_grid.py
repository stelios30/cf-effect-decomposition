import pytest
import numpy as np

from ced.actors.grid import GridWorldAction
from ced.envs.grid import GRID_COLS, GRID_ROWS, ITEM_GREEN, ITEM_NULL, ITEM_PINK, ITEM_YELLOW, Instruction, LOC_BOX_1, LOC_BOX_2


class TestState:
    @pytest.mark.parametrize("inst", [inst for inst in Instruction if inst not in [Instruction.pickup_pink, Instruction.pickup_green, Instruction.pickup_yellow]])
    def test_it_masks_out_pickup_instructions_in_sa_mode(self, inst, make_gw_state):
        state = make_gw_state(mode="a1", inst=inst)
        obs = state.to_obs(agent_id=0)
        for act in [GridWorldAction.pickup_pink, GridWorldAction.pickup_green, GridWorldAction.pickup_yellow]:
            assert not obs["mask"][act]

    @pytest.mark.parametrize("inst", [inst for inst in Instruction if inst not in [Instruction.pickup_pink, Instruction.pickup_green, Instruction.pickup_yellow]])
    def test_it_masks_out_pickup_instructions_in_ma_mode(self, inst, make_gw_state):
        state = make_gw_state(mode="a2", inst=(inst, inst))
        obs = state.to_obs(agent_id=1)
        for act in [GridWorldAction.pickup_pink, GridWorldAction.pickup_green, GridWorldAction.pickup_yellow]:
            assert not obs["mask"][act]

    @pytest.mark.parametrize("inst", list(Instruction))
    def test_it_masks_out_pickup_instructions_when_agent_is_not_standing_on_the_box(self, inst, make_gw_state):
        state = make_gw_state(mode="a1", inst=inst, agents_pos=[-1, -1])
        obs = state.to_obs(agent_id=0)
        for act in [GridWorldAction.pickup_pink, GridWorldAction.pickup_green, GridWorldAction.pickup_yellow]:
            assert not obs["mask"][act]

    @pytest.mark.parametrize("inst", [Instruction.pickup_pink, Instruction.pickup_green, Instruction.pickup_yellow])
    def test_it_does_not_mask_the_pickup_instructions_when_agent_is_standing_on_the_box(self, inst, make_gw_state):
        state = make_gw_state(mode="a1", inst=inst, agents_pos=[LOC_BOX_1, -1])
        obs = state.to_obs(agent_id=0)
        assert all(obs["mask"].tolist())


class TestEnvReset:
    @pytest.mark.parametrize(
        "inst,init_pos",
        [
            (Instruction.examine_box_1, 1 * GRID_COLS + GRID_COLS - 1), # one cell under the first box
            (Instruction.examine_box_2, 3 * GRID_COLS + GRID_COLS - 1), # one cell above the first box
        ],
    )
    def test_it_sets_agents_location_for_examine_instructions_in_sa_mode(self, inst, init_pos, make_gw_gym_env):
        env = make_gw_gym_env(mode="a1")
        obs, _ = env.reset(options={"inst": inst})
        assert obs["pos"] == init_pos
        assert obs["pos_teammate"] is None
        assert np.array_equal(obs["inst"], inst.one_hot())
        assert obs["inst_teammate"] is None

    def test_it_sets_agents_locations_for_examine_instructions_in_ma_mode(self, make_gw_gym_env):
        env = make_gw_gym_env(mode="a2")
        obs, _ = env.reset(options={"inst": Instruction.examine_box_2})

        # assert second agent (best-responder, the one we are training) always examines box 2
        assert obs["pos"] == 3 * GRID_COLS + GRID_COLS - 1
        assert np.array_equal(obs["inst"], Instruction.examine_box_2.one_hot())

        # assert first agent always examines box 1
        assert obs["pos_teammate"] == 1 * GRID_COLS + GRID_COLS - 1
        assert np.array_equal(obs["inst_teammate"], Instruction.examine_box_1.one_hot())

    @pytest.mark.parametrize(
        "inst",
        [inst for inst in Instruction if inst not in [Instruction.examine_box_1, Instruction.examine_box_2]],
    )
    def test_it_sets_agent_location_for_remaining_instructions_in_sa_mode(self, inst, make_gw_gym_env):
        env = make_gw_gym_env(mode="a1")
        obs, _ = env.reset(options={"inst": inst})
        assert obs["pos"] == LOC_BOX_1
        assert obs["pos_teammate"] is None
        assert np.array_equal(obs["inst"], inst.one_hot())
        assert obs["inst_teammate"] is None

        env = make_gw_gym_env(mode="a2")
        obs, _ = env.reset(options={"inst": inst})
        obs["pos"] == LOC_BOX_2
        obs["pos_teammate"] = LOC_BOX_1
        assert np.array_equal(obs["inst"], inst.one_hot())

    @pytest.mark.parametrize("inst,item", [
        (Instruction.pickup_pink, ITEM_PINK),
        (Instruction.pickup_green, ITEM_GREEN),
        (Instruction.pickup_yellow, ITEM_YELLOW),
    ])
    def test_a_box_has_an_item_specified_in_the_instruction_in_sa_mode(self, inst, item, make_gw_gym_env):
        env = make_gw_gym_env(mode="a1")
        _ = env.reset(options={"inst": inst})
        assert all(item in boxes_content for boxes_content in env.state.boxes_content)

    @pytest.mark.parametrize("inst,item", [
        ((Instruction.pickup_pink, Instruction.pickup_green), (ITEM_PINK, ITEM_GREEN)),
        ((Instruction.pickup_yellow, Instruction.pickup_pink), (ITEM_YELLOW, ITEM_PINK)),
        ((Instruction.pickup_yellow, Instruction.pickup_yellow), (ITEM_YELLOW, ITEM_YELLOW)),
    ])
    def test_a_box_has_an_item_specified_in_the_instruction_in_ma_mode(self, inst, item, make_gw_gym_env):
        env = make_gw_gym_env(mode="a2")
        _ = env.reset(options={"inst": inst})
        assert all(item[0] in boxes_content and item[1] in boxes_content for boxes_content in env.state.boxes_content)

    @pytest.mark.parametrize("inst,allowed_items", [
        (Instruction.examine_box_1, [ITEM_NULL]),
        (Instruction.examine_box_2, [ITEM_NULL]),
        (Instruction.pickup_pink, [ITEM_NULL]),
        (Instruction.pickup_green, [ITEM_NULL]),
        (Instruction.pickup_yellow, [ITEM_NULL]),
        (Instruction.goto_pink, [ITEM_PINK, ITEM_GREEN, ITEM_YELLOW]),
        (Instruction.goto_green, [ITEM_PINK, ITEM_GREEN, ITEM_YELLOW]),
        (Instruction.goto_yellow, [ITEM_PINK, ITEM_GREEN, ITEM_YELLOW]),
    ])
    def test_it_sets_agents_items_in_sa_mode(self, inst, allowed_items, make_gw_gym_env):
        env = make_gw_gym_env(mode="a1")
        obs, _ = env.reset(options={"inst": inst})
        assert obs["item"].size == 4
        assert obs["item"].argmax() in allowed_items
        assert obs["item_teammate"] is None

    @pytest.mark.parametrize("inst,allowed_items", [
        ((Instruction.examine_box_1, Instruction.examine_box_2), [ITEM_NULL]),
        ((Instruction.pickup_pink, Instruction.pickup_pink), [ITEM_NULL]),
        ((Instruction.pickup_green, Instruction.pickup_green), [ITEM_NULL]),
        ((Instruction.pickup_yellow, Instruction.pickup_yellow), [ITEM_NULL]),
        ((Instruction.goto_pink, Instruction.goto_pink), [ITEM_PINK, ITEM_GREEN, ITEM_YELLOW]),
        ((Instruction.goto_green, Instruction.goto_green), [ITEM_PINK, ITEM_GREEN, ITEM_YELLOW]),
        ((Instruction.goto_yellow, Instruction.goto_yellow), [ITEM_PINK, ITEM_GREEN, ITEM_YELLOW]),
    ])
    def test_it_sets_agents_items_in_ma_mode(self, inst, allowed_items, make_gw_gym_env):
        env = make_gw_gym_env(mode="a2")
        obs, _ = env.reset(options={"inst": inst})
        assert obs["item"].size == 4
        assert obs["item"].argmax() in allowed_items
        assert obs["item_teammate"].size == 4
        assert obs["item_teammate"].argmax() in allowed_items

    def test_it_initializes_state_for_ma_w_planner_mode(self, make_gw_env):
        env = make_gw_env(mode="multi-agent-w-planner")
        state = env.reset()

        loc_next_to_box_1, loc_next_to_box_2 = 1 * GRID_COLS + GRID_COLS - 1, 3 * GRID_COLS + GRID_COLS - 1

        assert state.stage == "planning"
        assert state.inst is None
        assert state.agents_pos == [loc_next_to_box_1, loc_next_to_box_2]
        assert state.agents_visitation == {0: [loc_next_to_box_1], 1: [loc_next_to_box_2]}
        assert state.boxes_pos == [LOC_BOX_1, LOC_BOX_2]
        assert state.agents_items == [ITEM_NULL, ITEM_NULL]


class TestEnvPositionStep:
    @pytest.mark.parametrize("init_pos,new_pos,action", [
        [1 * GRID_COLS + 0, 1 * GRID_COLS + 1, GridWorldAction.right], # successful move right
        [1 * GRID_COLS + GRID_COLS - 1, 1 * GRID_COLS + GRID_COLS - 1, GridWorldAction.right], # unsuccessful move right
        [1 * GRID_COLS + 0, 0 * GRID_COLS + 0, GridWorldAction.up], # successful move up
        [0 * GRID_COLS + 0, 0 * GRID_COLS + 0, GridWorldAction.up], # unsuccessful move up
        [1 * GRID_COLS + 1, 1 * GRID_COLS + 0, GridWorldAction.left], # successful move left
        [1 * GRID_COLS + 0, 1 * GRID_COLS + 0, GridWorldAction.left], # unsuccessful move left
        [1 * GRID_COLS + 0, 2 * GRID_COLS + 0, GridWorldAction.down], # successful move down
        [(GRID_ROWS - 1) * GRID_COLS + 0, (GRID_ROWS - 1) * GRID_COLS + 0, GridWorldAction.down] # unsuccessful move down
    ])
    def test_action_moves_agent_in_sa_mode(self, init_pos, new_pos, action, make_gw_gym_env, make_gw_state):
        state = make_gw_state(mode="a1", agents_pos=[init_pos, -1])
        env = make_gw_gym_env(mode="a1", init_state=state)
        obs, _, _, _, _ = env.step(action=action)
        assert obs["pos"] == new_pos

    @pytest.mark.parametrize("init_pos,new_pos,action", [
        [[12, 36], [13, 37], (GridWorldAction.right, GridWorldAction.right)], # successful move right
        [[1, 13], [0, 12], (GridWorldAction.left, GridWorldAction.left)], # successful move left
        [[12, 36], [24, 48], (GridWorldAction.down, GridWorldAction.down)], # successful move down
        [[12, 36], [0, 24], (GridWorldAction.up, GridWorldAction.up)], # successful move up
        [[13, 25], [13, 37], (GridWorldAction.down, GridWorldAction.down)], # first player collides into the second player, second player successfully moves down
        [[0, 12], [0, 0], (GridWorldAction.noop, GridWorldAction.up)], # second player collides into the first player, first player doesn't move
    ])
    def test_action_moves_agent_in_ma_mode(self, init_pos, new_pos, action, make_gw_gym_env, make_gw_state):
        state = make_gw_state(mode="a2", agents_pos=init_pos)
        env = make_gw_gym_env(mode="a2", init_state=state)
        obs, _, _, _, _ = env.step(action=action)
        assert obs["pos"] == new_pos[1]
        assert obs["pos_teammate"] == new_pos[0]

    @pytest.mark.parametrize("init_pos,new_pos,action", [
        [1 * GRID_COLS + 0, 1 * GRID_COLS + 1, GridWorldAction.right], # successful move right
        [1 * GRID_COLS + 0, 0 * GRID_COLS + 0, GridWorldAction.up], # successful move up
        [1 * GRID_COLS + 1, 1 * GRID_COLS + 0, GridWorldAction.left], # successful move left
        [1 * GRID_COLS + 0, 2 * GRID_COLS + 0, GridWorldAction.down], # successful move down
    ])
    def test_it_keeps_track_of_the_visited_states_in_sa_mode(self, init_pos, new_pos, action, make_gw_gym_env, make_gw_state):
        state = make_gw_state(mode="a1", agents_pos=[init_pos, -1], agents_visitation={0: [init_pos], 1: []})
        env = make_gw_gym_env(mode="a1", init_state=state)
        obs, _, _, _, _ = env.step(action=action)
        assert obs["pos"] == new_pos
        assert env.state.agents_visitation[0] == [init_pos, new_pos]

    @pytest.mark.parametrize("init_pos,new_pos,action", [
        [[12, 36], [13, 37], (GridWorldAction.right, GridWorldAction.right)], # successful move right
        [[1, 13], [0, 12], (GridWorldAction.left, GridWorldAction.left)], # successful move left
        [[12, 36], [24, 48], (GridWorldAction.down, GridWorldAction.down)], # successful move down
        [[12, 36], [0, 24], (GridWorldAction.up, GridWorldAction.up)], # successful move up
        [[13, 25], [13, 37], (GridWorldAction.down, GridWorldAction.down)], # first player collides into the second player, second player successfully moves down
        [[0, 12], [0, 0], (GridWorldAction.noop, GridWorldAction.up)], # second player collides into the first player, first player doesn't move
    ])
    def test_it_keeps_track_of_the_visited_states_in_ma_mode(self, init_pos, new_pos, action, make_gw_gym_env, make_gw_state):
        state = make_gw_state(mode="a2", agents_pos=init_pos, agents_visitation={0: [init_pos[0]], 1: [init_pos[1]]})
        env = make_gw_gym_env(mode="a2", init_state=state)
        obs, _, _, _, _ = env.step(action=action)
        assert obs["pos"] == new_pos[1]
        assert obs["pos_teammate"] == new_pos[0]
        assert set(env.state.agents_visitation[0]) == {init_pos[0], new_pos[0]}
        assert set(env.state.agents_visitation[1]) == {init_pos[1], new_pos[1]}


class TestEnvInventoryStep:
    @pytest.mark.parametrize("box_content,action,inventory", [
        ([ITEM_PINK, ITEM_YELLOW], GridWorldAction.pickup_yellow, ITEM_YELLOW),
        ([ITEM_GREEN, ITEM_YELLOW], GridWorldAction.pickup_green, ITEM_GREEN),
        ([ITEM_PINK, ITEM_GREEN], GridWorldAction.pickup_pink, ITEM_PINK),
    ])
    def test_inventory_changes_in_sa_mode(self, box_content, action, inventory, make_gw_gym_env, make_gw_state):
        state = make_gw_state(mode="a1", agents_pos=[LOC_BOX_1, -1], boxes_content=[box_content, []])
        env = make_gw_gym_env(mode="a1", init_state=state)
        _ = env.step(action=action)

        next_state = env.state
        assert next_state.agents_items[0] == inventory

    @pytest.mark.parametrize("boxes_content,action,inventory", [
        ([[ITEM_PINK, ITEM_YELLOW], [ITEM_GREEN, ITEM_YELLOW]], (GridWorldAction.pickup_yellow, GridWorldAction.pickup_green), [ITEM_YELLOW, ITEM_GREEN]),
        ([[ITEM_GREEN, ITEM_YELLOW], [ITEM_PINK, ITEM_GREEN]], (GridWorldAction.pickup_green, GridWorldAction.pickup_pink), [ITEM_GREEN, ITEM_PINK]),
        ([[ITEM_PINK, ITEM_GREEN], [ITEM_GREEN, ITEM_YELLOW]], (GridWorldAction.pickup_pink, GridWorldAction.pickup_yellow), [ITEM_PINK, ITEM_YELLOW]),
    ])
    def test_inventory_changes_in_ma_mode(self, boxes_content, action, inventory, make_gw_gym_env, make_gw_state):
        state = make_gw_state(mode="a2", agents_pos=[LOC_BOX_1, LOC_BOX_2], boxes_content=boxes_content)
        env = make_gw_gym_env(mode="a2", init_state=state)
        _ = env.step(action=action)

        next_state = env.state
        assert next_state.agents_items == inventory


class TestEnvWithPlannerStep:
    def test_it_sets_agents_instruction(self, make_gw_env, make_gw_planner_act):
        # get the initial state
        env = make_gw_env(mode="multi-agent-w-planner")

        state = env.reset()
        assert state.inst is None
        assert state.stage == "planning"

        # simulate planner's action
        action = make_gw_planner_act(
            inst=("goto pink", "goto yellow"),
            inst_context=[5, 6, 7, 0, 1],
        )
        next_state, reward = env.step(state=state, actions=[None, None, action], env_noise={})

        # check that the environment transitions correctly
        assert next_state.inst == ("goto pink", "goto yellow")
        assert next_state.inst_context == [5, 6, 7, 0, 1]
        assert reward == -0.4
        assert next_state.stage == "execution"

    @pytest.mark.parametrize("init_pos,new_pos,action", [
        [[12, 36], [13, 37], (GridWorldAction.right, GridWorldAction.right)], # successful move right
        [[1, 13], [0, 12], (GridWorldAction.left, GridWorldAction.left)], # successful move left
        [[12, 36], [24, 48], (GridWorldAction.down, GridWorldAction.down)], # successful move down
        [[13, 37], [1, 25], (GridWorldAction.up, GridWorldAction.up)], # successful move up
        [[13, 25], [13, 37], (GridWorldAction.down, GridWorldAction.down)], # first player collides into the second player, second player successfully moves down
        [[0, 12], [0, 0], (GridWorldAction.noop, GridWorldAction.up)], # second player collides into the first player, first player doesn't move
    ])
    def test_action_moves_agent_in_ma_w_planner_mode(self, init_pos, new_pos, action, make_gw_env, make_gw_state):
        state = make_gw_state(mode="multi-agent-w-planner", stage="execution", agents_pos=init_pos)
        env = make_gw_env(mode="multi-agent-w-planner")

        next_state, _ = env.step(state=state, actions=[*action, None], env_noise={})

        assert next_state.stage == "execution"
        assert next_state.agents_pos == new_pos

    @pytest.mark.parametrize("boxes_content,action,inventory,inst", [
        ([[ITEM_PINK, ITEM_YELLOW], [ITEM_GREEN, ITEM_YELLOW]], (GridWorldAction.pickup_yellow, GridWorldAction.pickup_green), [ITEM_YELLOW, ITEM_GREEN], (Instruction.pickup_yellow, Instruction.pickup_green)),
        ([[ITEM_GREEN, ITEM_YELLOW], [ITEM_PINK, ITEM_GREEN]], (GridWorldAction.pickup_green, GridWorldAction.pickup_pink), [ITEM_GREEN, ITEM_PINK], (Instruction.pickup_green, Instruction.pickup_pink)),
        ([[ITEM_PINK, ITEM_GREEN], [ITEM_GREEN, ITEM_YELLOW]], (GridWorldAction.pickup_pink, GridWorldAction.pickup_yellow), [ITEM_PINK, ITEM_YELLOW], (Instruction.pickup_pink, Instruction.pickup_yellow)),
    ])
    def test_action_changes_the_stage_once_pickup_instruction_is_completed(self, boxes_content, action, inventory, inst, make_gw_state, make_gw_env):
        state = make_gw_state(
            inst=inst,
            mode="multi-agent-w-planner",
            stage="execution",
            agents_pos=[LOC_BOX_1, LOC_BOX_2],
            boxes_content=boxes_content,
        )
        env = make_gw_env(mode="multi-agent-w-planner")

        next_state, _ = env.step(state=state, actions=[*action, None], env_noise={})

        assert next_state.stage == "planning"
        assert next_state.agents_items == inventory
