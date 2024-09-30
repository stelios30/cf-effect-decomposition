import pickle
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import typer
from dotenv import load_dotenv
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from typing_extensions import Annotated

from ced.actors.causal import CausalActor
from ced.actors.grid import (GridWorldAction, GridWorldActor, PlannerAction, PlannerActor,
                             Reporter, make_dqn_policy)
from ced.envs.grid import (GRID_COLS, GRID_ROWS, ITEM_GREEN, ITEM_PINK,
                           LOC_CORRIDORS, GridWorld, GridWorldGym,
                           GridWorldTrajectory, Instruction)
from ced.tools.algorithms import ase, reverse_sse, reverse_sse_icc, sse, tcfe
from ced.tools.llm import LLMConfig, LLMGateway
from ced.tools.noise import UniformNoiseModel
from ced.tools.order import Order
from ced.tools.utils import (configure_logger, decay_schedule, find_by_id,
                             sample_trajectories, to_device)

app = typer.Typer()


@app.command(help="Train an instruction-following agent.")
def train_ifa(
    artifacts_dir: Annotated[Path, typer.Option(help="Path to directory where artifacts are stored. There will be one subdirectory per random seed.", dir_okay=True, file_okay=False)],
    seed: Annotated[int, typer.Option(help="Random seed to use for reproducibility.")] = 55969672,
    instruction: Annotated[Optional[Instruction], typer.Option(help="Instruction to train the agent on. If `None` agent is trained to follow all instructions.", case_sensitive=False)] = None,
    mode: Annotated[Optional[str], typer.Option(help="Agent to train, either 'a1' or 'a2'")] = "a1",
    base_agent_path: Annotated[Path, typer.Option(help="Path to a pre-trained agent to further fine-tune.", dir_okay=False, file_okay=True)] = None,
    discount: Annotated[float, typer.Option(help="Discount factor.", min=0.0, max=1.0)] = 0.99,
    num_step_per_collect: Annotated[int, typer.Option(help="Number of new transitions to collect per each collect procedure.")] = 200,
    num_collect_per_epoch: Annotated[int, typer.Option(help="Number of collect procedures per each epoch.")] = 5,
    num_epochs: Annotated[int, typer.Option(help="Number of training epochs to run. Each epoch collects 2 * `num_step_per_epoch` transitions.")] = 100,
    lr: Annotated[float, typer.Option(help="Learning rate.", min=0.0, max=1.0)] = 5e-4,
    hidden_dim: Annotated[int, typer.Option(help="Size of the hidden layers for actor and critic networks.")] = 256,
    hidden_depth: Annotated[int, typer.Option(help="Number of hidden layers for actor and critic networks.")] = 2,
    estimation_step: Annotated[int, typer.Option(help="Number of steps used during n-step value estimation.")] = 5,
    target_update_freq: Annotated[int, typer.Option(help="Number of steps after which the target network is updated. Set to 1 when using Polyak averaging.")] = 1500,
    polyak_tau: Annotated[float, typer.Option(help="Target network update rate.", min=0.0, max=1.0)] = 1e-3,
    batch_size: Annotated[int, typer.Option(help="Number of transitions to sample for policy update.")] = 512,
    horizon: Annotated[int, typer.Option(help="Maximum number of episode steps.")] = 25,
    num_train_envs: Annotated[int, typer.Option(help="Number of parallel environments to use for training.")] = 8,
    eps_greedy: Annotated[float, typer.Option(help="Initial epsilon in the eps-greedy exploration. Decayed to 0.01.")] = 0.5,
    eps_greedy_min: Annotated[float, typer.Option(help="Minimum value of the eps-greedy exploration.")] = 0.05,
    eps_greedy_decay: Annotated[float, typer.Option(help="Decay ratio of the eps-greedy exploration.")] = 0.4,
    eps_greedy_decay_epochs: Annotated[int, typer.Option(help="Number of epochs after which the epsilon is decayed.")] = 500,
    device: Annotated[str, typer.Option(help="Device to use for training.")] = "cuda",
    replay_buffer_size: Annotated[int, typer.Option(help="Size of the replay buffer.")] = 1e4,
    parallelize: Annotated[bool, typer.Option(help="Indicates if environments should use parallelization. Typically disabled for debugging.")] = False,
    weighted_instruction_sampling: Annotated[bool, typer.Option(help="Indicates if we use a non-uniform instruction sampling during training.", is_flag=False)] = False,
    eval_on_instructions: Annotated[Optional[str], typer.Option(help="A comma-separated list of instructions to evaluate the agent on. Set to 'None' to evaluate on all instructions.")] = None,
):
    # === set random seed for reproducibility ===
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # === initialize the policy ===
    policy = make_dqn_policy(
        lr=lr,
        hidden_size=hidden_dim,
        hidden_depth=hidden_depth,
        n_estimation_step=estimation_step,
        target_update_freq=target_update_freq,
        num_pos_embeddings=GRID_COLS * GRID_ROWS,
        num_item_embeddings=4,
        tau=polyak_tau,
        discount=discount,
        device=device,
    )

    if base_agent_path is not None:
        checkpoint = torch.load(base_agent_path, map_location=device)["policy"]
        checkpoint = to_device(checkpoint, device=device)
        policy.load_state_dict(checkpoint.state_dict())

    # === initialize environments ===
    def _get_env_factory(inst: Optional[Instruction] = None):
        def _factory():
            env = GridWorld(mode=mode)
            env = GridWorldGym(env=env, instruction=inst if inst is not None else instruction, weighted_sampling=weighted_instruction_sampling)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=horizon)
            return env
        return _factory

    # initialize instructions to evaluate on
    eval_instructions = list(Instruction) if eval_on_instructions is None else [Instruction(inst.strip()) for inst in eval_on_instructions.split(",")]
    inst_envs = {inst.name: DummyVectorEnv([_get_env_factory(inst=inst)]) for inst in eval_instructions}
    [env.seed(seed) for env in inst_envs.values()]

    # === initialize trainer ===
    eps = decay_schedule(eps_greedy, min_value=eps_greedy_min, decay_ratio=eps_greedy_decay, max_steps=eps_greedy_decay_epochs)
    logger = configure_logger(log_dir=artifacts_dir)

    def _on_train_epoch_start(epochs: int, step_idx: int):
        policy.set_eps(eps[epochs - 1] if epochs <= len(eps) else eps[-1])

    def _on_save_checkpoint(epoch: int, env_step: int, gradient_step: int):
        torch.save({"policy": policy}, artifacts_dir / "policy.pt")

    # run the training loop for each requested instruction
    env_cls = SubprocVectorEnv if parallelize else DummyVectorEnv

    train_envs = env_cls([_get_env_factory(inst=instruction) for _ in range(num_train_envs)])
    train_envs.seed(seed)
    train_buffer = PrioritizedVectorReplayBuffer(replay_buffer_size, num_train_envs, alpha=0.6, beta=0.9)
    train_collector = Collector(policy, train_envs, buffer=train_buffer, exploration_noise=True)

    test_envs = env_cls([_get_env_factory(inst=instruction)])
    test_envs.seed(seed)
    test_collector = Collector(policy, test_envs)

    tb_writer = SummaryWriter(artifacts_dir / "tensorboard")
    tb_logger = TensorboardLogger(tb_writer, save_interval=1, train_interval=400)

    trainer = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=num_epochs,
        batch_size=batch_size,
        step_per_epoch=num_collect_per_epoch,
        step_per_collect=num_step_per_collect * num_train_envs,
        episode_per_test=1,
        update_per_step=1,
        repeat_per_collect=1,
        logger=tb_logger,
        save_checkpoint_fn=_on_save_checkpoint,
        train_fn=_on_train_epoch_start,
        test_in_train=False,
        verbose=False,
        show_progress=False,
    )

    # === training ===
    for epoch, epoch_stat, _ in trainer:
        log_message = f"train/rew={epoch_stat['rew']} "

        # evaluate the performance on each instruction
        for inst_name, env in inst_envs.items():
            collector = Collector(policy, env, exploration_noise=False)
            result = collector.collect(n_episode=25)
            log_message += f"instruction/{inst_name}/rew={result['rew']} "

        logger.bind(epoch=epoch).info(log_message)


@app.command(help="Evaluate an instruction-following agent, optionally recording videos.")
def evaluate_ifa(
    policy_path: Annotated[Path, typer.Option(help="Path to the policy to evaluate.", file_okay=True, dir_okay=False)],
    mode: Annotated[str, typer.Option(help="Mode of the environment to evaluate the policy in (a1, a2, multi-agent-w-planner).")] = "a1",
    n_episode: Annotated[int, typer.Option(help="Number of episodes to evaluate the policy on.")] = 100,
    render: Annotated[Optional[float], typer.Option(help="Render the environment during evaluation each `render` seconds.")] = 1,
    horizon: Annotated[int, typer.Option(help="Maximum number of episode steps.")] = 25,
    device: Annotated[str, typer.Option(help="Device to use for evaluation.")] = "cuda",
    instruction: Annotated[Optional[str], typer.Option(help="Instruction to evaluate on (use comma to indicate per-agent instructions when appropriate). If `None` evaluation uses all instructions.")] = None,
    record: Annotated[Optional[Path], typer.Option(help="Path to a directory where to store the evaluation videos. Set to `None` do disable recording.", dir_okay=True, file_okay=False)] = None,
):
    policy = torch.load(policy_path, map_location=device)["policy"].eval()
    policy = to_device(policy, device)

    instructions = [Instruction(i.strip()) for i in instruction.split(",")] if instruction is not None else list(Instruction)
    instructions = [tuple(instructions)] if len(instructions) == 2 else [instruction]

    render_mode = "rgb_array" if record is not None else "human"
    logger = configure_logger(log_epochs=False)

    for inst in instructions:
        inst_name = f"{inst[0].name}-{inst[1].name}" if isinstance(inst, Tuple) else inst.name

        # create the environment
        env = GridWorld(mode=mode)
        env = GridWorldGym(env=env, instruction=inst, render_mode=render_mode)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=horizon)
        if record:
            env = gym.wrappers.RecordVideo(env, video_folder=record, name_prefix=inst_name, episode_trigger=lambda e: e < n_episode)
        env = DummyVectorEnv([lambda: env])

        # collect the statistics across multiple episodes
        collector = Collector(policy, env)
        result = collector.collect(n_episode=n_episode, render=render if record is None else None)

        # log the results
        logger.info(f"{inst_name}/rew={result['rew']} {inst_name}/rew_std={result['rew_std']} {inst_name}/len={result['len']} {inst_name}/len_std={result['len_std']}")
        env.close()


@app.command(help="Generate the results for the Gridworld experiment.")
def run(
    results_dir: Annotated[Path, typer.Option(help="Path to directory where artifacts are stored, typically `results/grid`.", dir_okay=True, file_okay=False)],
    env_file: Annotated[Optional[Path], typer.Option(help="Path to the .env file defining environment variables.", dir_okay=False, file_okay=True)] = None,
    num_trajectories: Annotated[int, typer.Option(help="Number of trajectories to sample per phase.")] = 10,
    num_cf_samples: Annotated[int, typer.Option(help="Number of counterfactual samples to use for the analysis.")] = 100,
    num_cf_samples_cond: Annotated[int, typer.Option(help="Number of counterfactual samples to use for the analysis.")] = 20,
    horizon: Annotated[int, typer.Option(help="Maximum number of episode steps.")] = 25,
    device: Annotated[str, typer.Option(help="Device to use for running policy inference.")] = "cpu",
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility.")] = 5656992596,
):
    # === Helper functions ===
    def _load_actor(path: str, id: int, device: str) -> GridWorldActor:
        policy = torch.load(path, map_location=device)["policy"].eval()
        policy = to_device(policy, device)
        actor = GridWorldActor(id, policy)
        return actor

    def _sample_or_load_trajectories(path: Path, env: Optional[GridWorld] = None, agents: List[CausalActor] = [], num_trajectories: int = 50, rng: Optional[np.random.Generator] = None, filter: Optional[Callable] = None) -> List[GridWorldTrajectory]:
        if path.exists():
            print("Found sampled trajectories, loading from disk.")
            with open(path, "rb") as f: return pickle.load(f)

        trajectories = []
        penalties = {(-25., -25., -25., -25.): False, (-15., -25., -25., -25.): False, (-25., -15., -25., -25.): False, (-25., -25., -15., -25.): False, (-25., -25., -25., -15.): False}

        while len(trajectories) < num_trajectories:
            t = sample_trajectories(env, agents=agents, num_trajectories=1, rng=rng)[0]
            penalty = tuple(t.get_corridor_penalties(agent_id=1, corridor="pink"))

            if filter is not None and filter(t) or filter is None:
                if not all(penalties.values()) and penalties[penalty]:
                    # skip if we already have a trajectory with the same penalties, until we have one of each
                    continue

                t.id = len(trajectories)
                penalties[penalty] = True

                trajectories.append(t)
                print(f"Sampled a new trajectory with penalties: {t.get_corridor_penalties(agent_id=1, corridor='pink')}, remaining: {num_trajectories - len(trajectories)}")

        with open(path, "wb") as f: pickle.dump(trajectories, f)
        return trajectories

    def _calculate_or_load_tcfe(path: Path, phase: int, trajectories: List[GridWorldTrajectory], env: GridWorld, agents: List[CausalActor], rng: np.random.Generator, time_steps: List[int], interventions: List[Dict]):
        if (path / f"p{phase}_trajectories.csv").exists():
            print("Found calculated TCFE data-frame, loading from disk.")
            return pd.read_csv(path / f"p{phase}_trajectories.csv")

        seeds = rng.integers(1e5, 1e6, size=len(trajectories))
        values = [tcfe(trajectory=t, env=env, agents=agents, intervention=i, time_step=ts, num_cf_samples=num_cf_samples, seed=s)
                for t, i, ts, s in zip(trajectories, interventions, time_steps, seeds)]

        data = pd.DataFrame({"traj_id": [t.id for t in trajectories], "seed": seeds, "tcfe": values})
        data.to_csv(path / f"p{phase}_trajectories.csv", index=False)
        return data

    def _calculate_or_load_causal_quantities(path: Path, phase: int, trajectories: List[GridWorldTrajectory], env: GridWorld, agents: List[CausalActor], rng: np.random.Generator, time_steps: List[int], interventions: List[Dict], num_cf_samples: int = 100, calculate_sse_icc: bool = True):
        if (path / f"p{phase}_causal_quantities.csv").exists():
            print("Found calculated causal quantities data-frame, loading from disk.")
            return pd.read_csv(path / f"p{phase}_causal_quantities.csv")

        data = []
        seeds = rng.integers(1e5, 1e6, size=len(trajectories))

        for traj, time_step, intr, seed in zip(trajectories, time_steps, interventions, seeds):
            item = {"traj_id": traj.id, "seed": seed, "reverse_sse_icc": ""}

            print(f"Calculating TCFE for trajectory {traj.id}...")
            item["tcfe"] = tcfe(trajectory=traj, env=env, agents=agents, intervention=intr, time_step=time_step, num_cf_samples=num_cf_samples, seed=seed)

            print(f"Calculating total-ASE for trajectory {traj.id}...")
            item["ase_total"] = ase(trajectory=traj, env=env, agents=agents, intervention=intr, effect_agents=[a.id for a in agents], time_step=time_step, num_cf_samples=num_cf_samples, seed=seed)

            for agent in agents:
                print(f"Calculating ASE for trajectory {traj.id} and agent {agent.id}...")
                item[f"ase_{agent.id}"] = ase(trajectory=traj, env=env, agents=agents, intervention=intr, effect_agents=[agent.id], time_step=time_step, num_cf_samples=num_cf_samples, seed=seed)

            print(f"Calculating SSE for trajectory {traj.id}...")
            item["sse"] = sse(trajectory=traj, env=env, agents=agents, intervention=intr, time_step=time_step, num_cf_samples=num_cf_samples, seed=seed)

            print(f"Calculating Reverse-SSE for trajectory {traj.id}...")
            item["reverse_sse"] = reverse_sse(trajectory=traj, env=env, agents=agents, intervention=intr, time_step=time_step, num_cf_samples=num_cf_samples, seed=seed)

            if calculate_sse_icc:
                print(f"Saving intermediate dataset before calculating intrinsic causal contributions for trajectory {traj.id}...")
                pd.DataFrame(data + [item]).to_csv(path / f"p{phase}_causal_quantities.csv", index=False)

                print(f"Calculating Reverse-SSE intrinsic causal contributions for trajectory {traj.id}...")
                item["reverse_sse_icc"] = reverse_sse_icc(trajectory=traj, env=env, agents=agents, reverse_sse_value=item["reverse_sse"], intervention=intr, time_step=time_step,
                                                        num_cf_samples=num_cf_samples, num_cf_samples_cond=num_cf_samples_cond, seed=seed)

            print(f"Finished calculating causal quantities for trajectory {traj.id}.")
            data.append(item)

        data = pd.DataFrame(data)
        data.to_csv(path / f"p{phase}_causal_quantities.csv", index=False)
        return data

    def _is_trajectory_selected(t: GridWorldTrajectory) -> bool:
        include = True

        # ensure box 1 contains (pink, *) and box 2 contains (pink green)
        include = include and t.states[0].boxes_content[0][0] == ITEM_PINK
        include = include and t.states[0].boxes_content[1] == [ITEM_PINK, ITEM_GREEN]

        # ensure planner instructs the actors to pickup pink items
        planner_actions = [planner_act for _, _, planner_act in t.actions if planner_act is not None]
        include = include and any([act.inst == (Instruction.pickup_pink, Instruction.pickup_pink) for act in planner_actions])

        # ensure the second actor follow the pink corridor
        include = include and set(LOC_CORRIDORS["pink"]).issubset(set(t.states[-1].agents_visitation[1]))

        # ensure the second actor received high penalties (HHHH, MHHH, HMHH, HHMH, HHHM) where H=highest and M=medium
        target = [[-25, -25, -25, -25], [-15, -25, -25, -25], [-25, -15, -25, -25], [-25, -25, -15, -25], [-25, -25, -25, -15]]
        penalties = t.get_corridor_penalties(agent_id=1, corridor="pink")
        include = include and any(penalties == t for t in target)

        return include

    # === Prepare for the main experiment ===
    if env_file is not None:
        load_dotenv(dotenv_path=env_file)

    print(f"Running experiment for seed '{seed}'")
    results_dir_seed = results_dir / str(seed)
    results_dir_seed.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(seed)
    time_start = time.time()

    # construct actors
    a1 = _load_actor(results_dir / "a1_policy.pt", id=0, device=device)
    a2 = _load_actor(results_dir / f"a2_policy.pt", id=1, device=device)

    gateway = LLMGateway(config=LLMConfig())
    planner = PlannerActor(id=2, gateway=gateway, reporter=Reporter())

    agents = [a1, a2, planner]

    # construct environments
    env_original = GridWorld(mode="multi-agent-w-planner", init_mode="analysis", horizon=horizon)
    env_analysis = GridWorld(
        mode="multi-agent-w-planner",
        horizon=horizon,
        act_noise_model=UniformNoiseModel(order=Order(list(range(len(GridWorldAction))))),
        env_noise_model=UniformNoiseModel(order=Order(list(range(3)))),
    )

    # === Sample the required number of trajectories ===
    print(f"Sampling (or loading) {num_trajectories} for analysis...")
    trajectories = _sample_or_load_trajectories(results_dir / f"trajectories.pkl", env=env_original, agents=agents, num_trajectories=num_trajectories, rng=rng, filter=_is_trajectory_selected)

    # === Intervene on the A2's pickup action ===
    intervention = {1: GridWorldAction.pickup_green}
    time_step = 3

    # calculate TCFE of intervening on the A2's pickup action, for all trajectories
    print("Calculating TCFE of intervening on A2's pickup action and setting it to `pickup green`...")
    p4_tcfe = _calculate_or_load_tcfe(results_dir, phase=4, trajectories=trajectories, env=env_analysis, agents=agents, rng=rng,
                                                        time_steps=[time_step] * len(trajectories), interventions=[intervention] * len(trajectories))

    # calculate causal quantities for a trajectory with highest TCFE
    trajectory = find_by_id(trajectories, id=p4_tcfe.iloc[p4_tcfe.tcfe.argmax()].traj_id)
    _ = _calculate_or_load_causal_quantities(results_dir_seed, phase=4, trajectories=[trajectory], env=env_analysis, agents=agents, rng=rng,
                                            time_steps=[time_step], interventions=[intervention], num_cf_samples=num_cf_samples,
                                            calculate_sse_icc=True)

    # === Intervene on the Planner's action ===
    intervention = {2: PlannerAction(inst=(Instruction.pickup_pink.value, Instruction.pickup_green.value))}
    time_step = 2

    # calculate TCFE of intervening on the planner's pickup action for A2, for all trajectories
    print("Calculating TCFE of intervening on planners's pickup action for A2 and setting it to `pickup green`...")
    p3_tcfe = _calculate_or_load_tcfe(results_dir, phase=3, trajectories=trajectories, env=env_analysis, agents=agents, rng=rng,
                                                        time_steps=[time_step] * len(trajectories), interventions=[intervention] * len(trajectories))

    # calculate causal quantities for a trajectory with highest TCFE
    trajectory = find_by_id(trajectories, id=p3_tcfe.iloc[p3_tcfe.tcfe.argmax()].traj_id)
    _ = _calculate_or_load_causal_quantities(results_dir_seed, phase=3, trajectories=[trajectory], env=env_analysis, agents=agents, rng=rng,
                                            time_steps=[time_step], interventions=[intervention], num_cf_samples=num_cf_samples,
                                            calculate_sse_icc=False)

    print(f"Experiment finished. Time elapsed: {time.time() - time_start} seconds.")

if __name__ == "__main__":
    app()
