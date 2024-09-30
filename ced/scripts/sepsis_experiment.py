import itertools
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tqdm
import typer
from joblib import Parallel, delayed
from numpy.random import Generator
from rich import print
from typing_extensions import Annotated

from ced.actors.sepsis import AIActor, ClinicianActor
from ced.envs.sepsis import Sepsis, SepsisAction, SepsisTrajectory, State
from ced.tools.algorithms import ase, ase_sv, reverse_sse, sse, tcfe, reverse_sse_icc, reverse_sse_variance
from ced.tools.noise import SepsisStateUniformNoiseModel, UniformNoiseModel
from ced.tools.order import Order
from ced.tools.utils import find_by_id, sample_trajectories


def main(
    # experiment parameters
    seeds: Annotated[List[int], typer.Argument(help="List of random seeds to use for repeated experiment runs.")],
    artifacts_dir: Annotated[Path, typer.Option(help="Path to directory where artifacts are stored. There will be one subdirectory per random seed.", dir_okay=True, file_okay=False)],
    mdp_path: Annotated[Path, typer.Option(help="Path to exported MDP dynamics (see `notebooks/sepsis/learn_sepsis_mdp.ipynb` notebook).", dir_okay=False, file_okay=True)],
    cl_policy_path: Annotated[Path, typer.Option(help="Path to exported clinician policy (see `notebooks/sepsis/learn_sepsis_actors.ipynb` notebook).", dir_okay=False, file_okay=True)],
    ai_policy_path: Annotated[Path, typer.Option(help="Path to exported AI policy (see `notebooks/sepsis/learn_sepsis_actors.ipynb` notebook).", dir_okay=False, file_okay=True)],
    num_jobs: Annotated[int, typer.Option(help="Number of jobs to execute in parallel.")] = -1,
    # evaluation parameters
    tcfe_threshold: Annotated[float, typer.Option(help="Minimum TCFE an intervention needs to have to be considered for analysis.")] = 0.8,
    num_trajectories: Annotated[int, typer.Option(help="Number of trajectories to sample for analysis, per trust level.")] = 100,
    round_difference: Annotated[int, typer.Option(help="Round difference to be considered for calculating reverse SSE contribution scores.")] = 10,
    reverse_sse_threshold: Annotated[float, typer.Option(help="Minimum absolute value of reverse SSE an intervention needs to have in order to be considered for calculating contribution scores.")] = 0.1,
    reverse_sse_variance_threshold: Annotated[float, typer.Option(help="Minimum reverse SSE variance an intervention needs to have in order to be considered for calculating contribution scores.")] = 0.01,
    # algorithms parameters
    num_cf_samples: Annotated[int, typer.Option(help="Number of counterfactual samples to draw when calculating counterfactual effects.")] = 100,
    num_cf_samples_cond: Annotated[int, typer.Option(help="Number of counterfactual samples to draw when calculating reverse SSE contribution scores.")] = 20,
    # envrironment parameters
    trust_values: Annotated[str, typer.Option(help="Comma-separated list of trust values to use for clinician's policy.")] = "0.0,0.2,0.4,0.6,0.8,1.0",
    max_horizon: Annotated[int, typer.Option(help="Maximum horizon of sampled trajectories.")] = 40,
):
    time_start = time.time()

    for seed in seeds:
        # sets up training artifacts and seed
        trust_levels = [float(t) for t in trust_values.split(",")]

        artifacts = artifacts_dir / str(seed)
        artifacts.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(seed)
        print(f"Running experiment for seed {seed}.")

        # loads or samples trajectories and counterfactuals, for each trust level
        trajectories, counterfactuals = _enumerate_or_load_counterfactuals(artifacts_dir=artifacts, num_trajectories=num_trajectories, trust_levels=trust_levels, ai_policy_path=ai_policy_path,
                                                                           cl_policy_path=cl_policy_path, mdp_path=mdp_path, rng=rng, max_horizon=max_horizon, num_cf_samples=num_cf_samples)

        # select only those interventions with specified TCFE and whose actions were taken at least two time steps before the end of the trajectory
        counterfactuals = counterfactuals.loc[np.where((counterfactuals.tcfe >= tcfe_threshold))]
        print(f"Selected {len(counterfactuals)} counterfactuals with TCFE >= {tcfe_threshold} for analysis.")

        # calculate causal quantities for all selected interventions
        counterfactuals = _calculate_causal_quantities(artifacts_dir=artifacts, counterfactuals=counterfactuals, trajectories=trajectories, num_cf_samples=num_cf_samples,
                                                       mdp_path=mdp_path, ai_policy_path=ai_policy_path, cl_policy_path=cl_policy_path, max_horizon=max_horizon, num_jobs=num_jobs)

        # calculate round difference for all selected interventions (where each round consists of two consecutive time steps, during which first the AI and then the clinician take an action)
        counterfactuals = _calculate_round_difference(artifacts_dir=artifacts, counterfactuals=counterfactuals, trajectories=trajectories)

        # calculate total reverse state-specific effect variance
        counterfactuals = _calculate_total_variance(artifacts_dir=artifacts, ai_policy_path=ai_policy_path, cl_policy_path=cl_policy_path, mdp_path=mdp_path, counterfactuals=counterfactuals,
                                                    trajectories=trajectories, max_horizon=max_horizon, num_cf_samples=num_cf_samples, num_jobs=num_jobs)

        # select only those interventions with specified round difference
        counterfactuals = counterfactuals.loc[np.where((counterfactuals.round_diff == round_difference))].reset_index(drop=True)
        # select only those interventions with absolute reverse SSE value above the specified threshold
        counterfactuals = counterfactuals.loc[np.where((abs(counterfactuals.reverse_sse) >= reverse_sse_threshold))].reset_index(drop=True)
        # select only those interventions with reverse SSE variance above the specified threshold
        counterfactuals = counterfactuals.loc[np.where((counterfactuals.total_rsse_variance >= reverse_sse_variance_threshold))]

        print(f"Selected {len(counterfactuals)} counterfactuals with round difference = {round_difference}, absolute reverse SSE value >= {reverse_sse_threshold} and reverse SSE variance >= {reverse_sse_variance_threshold} for calculating contribution scores.")

        # calculate reverse SSE contribution scores for a specific round difference
        counterfactuals = _calculate_rsse_contribution_scores(artifacts_dir=artifacts, ai_policy_path=ai_policy_path, cl_policy_path=cl_policy_path, mdp_path=mdp_path, counterfactuals=counterfactuals,
                                                              trajectories=trajectories, round_difference=round_difference, max_horizon=max_horizon, num_cf_samples_cond=num_cf_samples_cond, num_cf_samples=num_cf_samples, num_jobs=num_jobs)


    print(f"Experiment finished. Time elapsed: {time.time() - time_start} seconds.")


def _enumerate_or_load_counterfactuals(
    artifacts_dir: Path, num_trajectories: int, trust_levels: List[float],
    ai_policy_path: str, cl_policy_path: str, mdp_path: str,
    max_horizon: int, num_cf_samples: int, rng: Generator,
) -> Tuple[List[SepsisTrajectory], pd.DataFrame]:
    trajectories_pkl_path = artifacts_dir / "trajectories.pkl"
    trajectories_txt_path = artifacts_dir / "trajectories.txt"
    counterfactuals_path = artifacts_dir / "counterfactuals.csv"

    # if we have already sampled trajectories and counterfactuals, load them from disk
    if trajectories_pkl_path.exists() and trajectories_txt_path.exists() and counterfactuals_path.exists():
        with open(trajectories_pkl_path, "rb") as f:
            trajectories, counterfactuals = pickle.load(f), pd.read_csv(counterfactuals_path)
            print(f"Loaded {len(trajectories)} trajectories from '{trajectories_pkl_path}'.")
            print(f"Loaded {len(counterfactuals)} counterfactuals from '{counterfactuals_path}'.")
            return trajectories, counterfactuals

    # for our analysis, we rely on a noise-monotonic simulator
    print(f"Initializing noise-monotonic sepsis simulator.")
    act_noise = UniformNoiseModel(order=Order([i for i in range(SepsisAction.NUM_FULL)]))
    env_noise = SepsisStateUniformNoiseModel(order=Order([i for i in range(State.NUM_TOTAL)]))
    env = Sepsis(dynamics=mdp_path, max_horizon=max_horizon, act_noise_model=act_noise, env_noise_model=env_noise, turn_based=True)

    # helper function that samples trajectories and enumerates counterfactuals for a given trust level
    def _calculate(trust_level: int, trust_level_id: int, seed: int):
        curr_trajectories, curr_counterfactuals = [], []
        seed_rng = np.random.default_rng(seed)

        # creates AI and clinician actors with the target trust level
        ai_agent = AIActor(id=0, policy=ai_policy_path, rng=seed_rng)
        cl_agent = ClinicianActor(id=1, policy=cl_policy_path, trust=trust_level, rng=seed_rng)
        agents = [ai_agent, cl_agent]

        # for trajectory sampling, we rely on the original simulator
        env_simulator = Sepsis(transition_probabilities="./assets/sepsis/sepsis_transition_probs_original.json", max_horizon=max_horizon, turn_based=True)

        # samples trajectories from the simulator
        curr_trajectories = sample_trajectories(env=env_simulator, agents=agents, num_trajectories=num_trajectories, rng=seed_rng, outcome_target=0.0)
        print(f"Sampled {len(curr_trajectories)} failed trajectories for clinician's trust level of {trust_level}.")

        # update identifier of trajectories to be unique across trust levels
        for t in curr_trajectories: t.id += trust_level_id * num_trajectories

        # remove trajectories for which the approximated distribution is not defined
        with open(mdp_path, "rb") as f: dynamics = pickle.load(f)["transition_matrix"]
        filtered = []
        for t in curr_trajectories:
            include = True
            for time_step in range(t.horizon - 1):
                state, action, next_state = t.states[time_step], t.actions[time_step][t.states[time_step].player], t.states[time_step + 1]
                if state.player == 1:
                    # only after the clinician's action we transition to the next state
                    action = action if action < SepsisAction.NUM_TOTAL else state.act_ai
                    include = include and dynamics[action][state.index][next_state.index] > 0.0
            if include: filtered.append(t)
        curr_trajectories = filtered
        print(f"Skips analysis for {num_trajectories - len(curr_trajectories)} trajectories for which the approximated distribution was not defined.")

        # enumerate counterfactuals for each trajectory, considering every alternative action of an active agent for all time-steps
        candidates = []
        for traj in tqdm.tqdm(curr_trajectories, desc="Enumerating possible interventions"):
            # skip trajectories which are too short
            if traj.horizon < 3:
                print(f"Skipping trajectory {traj.id} because of its horizon of {traj.horizon}")
                continue

            # consider all remaining time-steps and active agents
            for time_step in range(0, traj.horizon - 2):
                agent = agents[traj.states[time_step].player]
                act_taken = traj.actions[time_step][agent.id]
                act_alternative = [act for act in env.get_available_actions(traj.states[time_step], agent.id) if act != act_taken]
                for act in act_alternative: candidates.append({"traj": traj, "agent_id": agent.id, "time_step": time_step, "act": act})

        # calculate TCFE for each candidate
        tcfe_seeds = seed_rng.integers(1e5, size=len(candidates))
        tcfe_values = [
            tcfe(trajectory=c["traj"], intervention={c["agent_id"]: c["act"]}, time_step=c["time_step"], env=env, agents=agents, seed=tcfe_seeds[i], outcome_target=1.0, num_cf_samples=num_cf_samples)
            for i, c in enumerate(tqdm.tqdm(candidates, desc="Calculating TCFE values"))]

        # persist counterfactuals
        for c, tcfe_value, s in zip(candidates, tcfe_values, tcfe_seeds):
            curr_counterfactuals.append({"traj_id": c["traj"].id, "agent_id": c["agent_id"], "time_step": c["time_step"], "alternative": c["act"], "trust_level": trust_level, "tcfe": tcfe_value, "seed": s})

        return curr_trajectories, curr_counterfactuals

    # runs enumeration of counterfactuals in parallel for each trust level
    seeds = rng.integers(1e5, size=len(trust_levels))
    results = Parallel(n_jobs=8)(delayed(_calculate)(trust_level, trust_level_id, seed) for trust_level_id, (trust_level, seed) in enumerate(zip(trust_levels, seeds)))
    trajectories = list(itertools.chain(*list(map(lambda r: r[0], results))))
    counterfactuals = list(itertools.chain(*list(map(lambda r: r[1], results))))

    # saves trajectories to disk
    with open(trajectories_pkl_path, "wb") as f:
        pickle.dump(trajectories, f)
        print(f"Saved pickled trajectories under '{trajectories_pkl_path}'.")
    with open(trajectories_txt_path, "w") as f:
        f.write("\n".join([t.render() for t in trajectories]))
        print(f"Saved human-readable trajectories under '{trajectories_txt_path}'.")

    # saves counterfactuals to disk
    df = pd.DataFrame(counterfactuals)
    df.to_csv(counterfactuals_path, index=False)
    print(f"Saved {len(df)} counterfactuals to file '{counterfactuals_path}'.")

    return trajectories, df


def _calculate_causal_quantities(
    artifacts_dir: Path, ai_policy_path: str, cl_policy_path: str, mdp_path: str,
    counterfactuals: pd.DataFrame, trajectories: List[SepsisTrajectory],
    num_cf_samples: int, max_horizon: int, num_jobs: int,
) -> pd.DataFrame:
    counterfactuals_w_quantities_path = artifacts_dir / "counterfactuals_w_quantities.csv"

    if counterfactuals_w_quantities_path.exists():
        df = pd.read_csv(counterfactuals_w_quantities_path)
        print(f"Loaded {len(df)} counterfactuals with calculated ASE, SSE and r-SSE quantities from {counterfactuals_w_quantities_path}.")
        return df

    # initialize a noise-monotonic sepsis simulator used for analysis
    print(f"Initializing noise-monotonic sepsis simulator.")
    act_noise = UniformNoiseModel(order=Order([i for i in range(SepsisAction.NUM_FULL)]))
    env_noise = SepsisStateUniformNoiseModel(order=Order([i for i in range(State.NUM_TOTAL)]))
    env = Sepsis(dynamics=mdp_path, max_horizon=max_horizon, act_noise_model=act_noise, env_noise_model=env_noise, turn_based=True)

    # helper function that calculates causal quantities for a chunk of counterfactuals
    def _calculate(counterfactuals: pd.DataFrame):
        df_items = []

        for intr in tqdm.tqdm(counterfactuals.itertuples(), desc="Calculating causal quantities for each selected counterfactual"):
            item = {"traj_id": intr.traj_id, "agent_id": intr.agent_id, "time_step": intr.time_step, "alternative": intr.alternative, "trust_level": intr.trust_level, "tcfe": intr.tcfe, "seed": intr.seed}
            trajectory = find_by_id(trajectories, intr.traj_id)

            # creates AI and clinician actors with a target trust level
            ai_agent = AIActor(id=0, policy=ai_policy_path)
            cl_agent = ClinicianActor(id=1, policy=cl_policy_path, trust=intr.trust_level)
            agents = [ai_agent, cl_agent]

            # calculates cl-ASE
            effect_agent = [cl_agent.id]
            item["ase_cl"] = ase(trajectory=trajectory, env=env, agents=agents, intervention={intr.agent_id: intr.alternative}, outcome_target=1.0,
                                           effect_agents=effect_agent, time_step=intr.time_step, num_cf_samples=num_cf_samples, seed=intr.seed)

            # calculates ai-ASE
            effect_agent = [ai_agent.id]
            item["ase_ai"] = ase(trajectory=trajectory, env=env, agents=agents, intervention={intr.agent_id: intr.alternative}, outcome_target=1.0,
                                           effect_agents=effect_agent, time_step=intr.time_step, num_cf_samples=num_cf_samples, seed=intr.seed)

            # calculates the total ASE for this intervention
            effect_agents = [ai_agent.id, cl_agent.id]
            item["ase_total"] = ase(trajectory=trajectory, env=env, agents=agents, intervention={intr.agent_id: intr.alternative}, outcome_target=1.0,
                                    effect_agents=effect_agents, time_step=intr.time_step, num_cf_samples=num_cf_samples, seed=intr.seed)

            # calculate contribution scores for this intervention
            ase_sv_values = ase_sv(trajectory=trajectory, env=env, agents=agents, intervention={intr.agent_id: intr.alternative}, outcome_target=1.0,
                                   time_step=intr.time_step, num_cf_samples=num_cf_samples, seed=intr.seed)

            for agent_id, agent_sv_value in ase_sv_values.items():
                agent_sv_label = "ase_sv_ai" if agent_id == ai_agent.id else "ase_sv_cl"
                item[agent_sv_label] = agent_sv_value

            # calculate state-specific effect
            item["sse"] = sse(trajectory=trajectory, env=env, agents=agents, intervention={intr.agent_id: intr.alternative}, time_step=intr.time_step, num_cf_samples=num_cf_samples, outcome_target=1.0, seed=intr.seed)

            # calculate reverse state-specific effect
            item["reverse_sse"] = reverse_sse(trajectory=trajectory, env=env, agents=agents, intervention={intr.agent_id: intr.alternative}, time_step=intr.time_step, num_cf_samples=num_cf_samples, outcome_target=1.0, seed=intr.seed)

            # persists item, if the decomposition holds
            assert np.isclose(item["ase_sv_cl"] + item["ase_sv_ai"] - item["reverse_sse"], item["tcfe"]).all(), "Decomposition failed (ase_sv_ai + ase_sv_cl - rsse) != tcfe"
            df_items.append(item)

        return df_items

    # runs calculations in parallel, for a chunked dataset
    chunk_size = 500
    chunks = [counterfactuals[i:i + chunk_size] for i in range(0, counterfactuals.shape[0], chunk_size)]
    items = Parallel(n_jobs=num_jobs)(delayed(_calculate)(df) for df in chunks)
    items = itertools.chain(*items)

    # saves calculated values to disk
    df = pd.DataFrame(items)
    df.to_csv(counterfactuals_w_quantities_path, index=False)
    print(f"Saved {len(df)} counterfactuals with ASE, SSE and r-SSE values to '{counterfactuals_w_quantities_path}'.")
    return df


def _calculate_round_difference(
    artifacts_dir: Path, counterfactuals: pd.DataFrame, trajectories: List[SepsisTrajectory],
)-> pd.DataFrame:
    round_difference_path = artifacts_dir / "counterfactuals_w_round_difference.csv"

    if round_difference_path.exists():
        df = pd.read_csv(round_difference_path)
        print(f"Loaded {len(df)} counterfactuals with causal quantities and round difference from {round_difference_path}.")
        return df

    # calculate round difference and add the column to the dataframe
    df = counterfactuals.copy()
    df.loc[:, "round_diff"] = (df["traj_id"].apply(lambda id: find_by_id(trajectories, id).horizon) // 2) -  (df["time_step"] // 2)

    df.to_csv(round_difference_path, index=False)
    print(f"Saved {len(df)} counterfactuals with round difference to '{round_difference_path}'.")
    return df


def _calculate_total_variance(
    artifacts_dir: Path, ai_policy_path: str, cl_policy_path: str, mdp_path: str,
    counterfactuals: pd.DataFrame, trajectories: List[SepsisTrajectory],
    max_horizon: int, num_cf_samples: int, num_jobs: int,
)-> pd.DataFrame:
    total_variance_path = artifacts_dir / f"total_rsse_variance.csv"

    if total_variance_path.exists():
        df = pd.read_csv(total_variance_path)
        print(f"Loaded {len(df)} counterfactuals with total reverse SSE variance from {total_variance_path}.")
        return df

    # initialize a noise-monotonic sepsis simulator used for analysis
    print(f"Initializing noise-monotonic sepsis simulator.")
    act_noise = UniformNoiseModel(order=Order([i for i in range(SepsisAction.NUM_FULL)]))
    env_noise = SepsisStateUniformNoiseModel(order=Order([i for i in range(State.NUM_TOTAL)]))
    env = Sepsis(dynamics=mdp_path, max_horizon=max_horizon, act_noise_model=act_noise, env_noise_model=env_noise, turn_based=True)

    # helper function that calculates total reverse SSE variance for a chunk of counterfactuals
    def _calculate(counterfactuals: pd.DataFrame):
        df_items = []

        for intr in tqdm.tqdm(counterfactuals.itertuples(), desc="Calculating total reverse SSE variance for each selected counterfactual"):
            item = {"traj_id": intr.traj_id, "agent_id": intr.agent_id, "time_step": intr.time_step, "alternative": intr.alternative, "trust_level": intr.trust_level, "tcfe": intr.tcfe, "reverse_sse": intr.reverse_sse, "round_diff": intr.round_diff, "seed": intr.seed}
            trajectory = find_by_id(trajectories, intr.traj_id)

            # creates AI and clinician actors with a target trust level
            ai_agent = AIActor(id=0, policy=ai_policy_path)
            cl_agent = ClinicianActor(id=1, policy=cl_policy_path, trust=intr.trust_level)
            agents = [ai_agent, cl_agent]

            # calculate the total reverse SSE variance
            item["total_rsse_variance"] = reverse_sse_variance(trajectory=trajectory, env=env, agents=agents, intervention={intr.agent_id: intr.alternative}, time_step=intr.time_step, var_time_step=0, num_cf_samples_cond=1, num_cf_samples=num_cf_samples, seed=intr.seed)

            df_items.append(item)

        return df_items

    # runs calculations in parallel, for a chunked dataset
    chunk_size = 64
    chunks = [counterfactuals[i:i + chunk_size] for i in range(0, counterfactuals.shape[0], chunk_size)]
    items = Parallel(n_jobs=num_jobs)(delayed(_calculate)(df) for df in chunks)
    items = itertools.chain(*items)

    # saves calculated values to disk
    df = pd.DataFrame(items)
    df.to_csv(total_variance_path, index=False)
    print(f"Saved {len(df)} counterfactuals with total reverse SSE variance to '{total_variance_path}'.")
    return df


def _calculate_rsse_contribution_scores(
    artifacts_dir: Path, ai_policy_path: str, cl_policy_path: str, mdp_path: str,
    counterfactuals: pd.DataFrame, trajectories: List[SepsisTrajectory], round_difference: int,
    max_horizon: int, num_cf_samples_cond: int, num_cf_samples: int, num_jobs: int,
)-> pd.DataFrame:
    rsse_contribution_scores_path = artifacts_dir / f"reverse_sse_contribution_scores{round_difference}.csv"

    if rsse_contribution_scores_path.exists():
        df = pd.read_csv(rsse_contribution_scores_path)
        print(f"Loaded {len(df)} counterfactuals with reverse SSE contribution scores for round difference of {round_difference} from {rsse_contribution_scores_path}.")
        return df

    # initialize a noise-monotonic sepsis simulator used for analysis
    print(f"Initializing noise-monotonic sepsis simulator.")
    act_noise = UniformNoiseModel(order=Order([i for i in range(SepsisAction.NUM_FULL)]))
    env_noise = SepsisStateUniformNoiseModel(order=Order([i for i in range(State.NUM_TOTAL)]))
    env = Sepsis(dynamics=mdp_path, max_horizon=max_horizon, act_noise_model=act_noise, env_noise_model=env_noise, turn_based=True)

    # helper function that calculates reverse SSE contribution scores for a chunk of counterfactuals
    def _calculate(counterfactuals: pd.DataFrame):
        df_items = []

        for intr in tqdm.tqdm(counterfactuals.itertuples(), desc="Calculating reverse SSE contribution scores for each selected counterfactual"):
            item = {"traj_id": intr.traj_id, "agent_id": intr.agent_id, "time_step": intr.time_step, "alternative": intr.alternative, "trust_level": intr.trust_level, "tcfe": intr.tcfe, "reverse_sse": intr.reverse_sse, "total_rsse_variance": intr.total_rsse_variance, "round_diff": intr.round_diff, "seed": intr.seed}
            trajectory = find_by_id(trajectories, intr.traj_id)

            # creates AI and clinician actors with a target trust level
            ai_agent = AIActor(id=0, policy=ai_policy_path)
            cl_agent = ClinicianActor(id=1, policy=cl_policy_path, trust=intr.trust_level)
            agents = [ai_agent, cl_agent]

            # calculate reverse state-specific effect contribution scores
            item["reverse_sse_contribution"] = reverse_sse_icc(trajectory=trajectory, env=env, agents=agents, intervention={intr.agent_id: intr.alternative}, time_step=intr.time_step, num_cf_samples_cond=num_cf_samples_cond, num_cf_samples=num_cf_samples, outcome_target=1.0, reverse_sse_value=item["reverse_sse"], seed=intr.seed)

            df_items.append(item)

        return df_items

    # runs calculations in parallel, for a chunked dataset
    chunk_size = 8
    chunks = [counterfactuals[i:i + chunk_size] for i in range(0, counterfactuals.shape[0], chunk_size)]
    items = Parallel(n_jobs=num_jobs)(delayed(_calculate)(df) for df in chunks)
    items = itertools.chain(*items)

    # saves calculated values to disk
    df = pd.DataFrame(items)
    df.to_csv(rsse_contribution_scores_path, index=False)
    print(f"Saved {len(df)} counterfactuals with reverse SSE contribution scores for round difference of {round_difference} to '{rsse_contribution_scores_path}'.")
    return df


if __name__ == "__main__":
    typer.run(main)
