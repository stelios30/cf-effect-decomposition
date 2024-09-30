import itertools
from math import factorial, isclose
from typing import Dict, List, Optional

import numpy as np
import tqdm

from ced.envs.causal import CausalActor, CausalEnv, Trajectory


def tcfe(trajectory: Trajectory, env: CausalEnv, agents: List[CausalActor], intervention: Dict[int, int], time_step: int, num_cf_samples: int = 100, outcome_target: Optional[float] = None, seed: Optional[int] = None) -> float:
    """Implements the total counterfactual effect.
    Args:
        trajectory (Trajectory): The trajectory to calculate the total counterfactual effect for.
        env (CausalEnv): Instance of the environment to sample counterfactuals from.
        agents (List[CausalActor]): List of agents acting in the environment.
        intervention (Dict[int, int]): The intervention to apply, with keys indicating agent's id and values indicating agent's action.
        time_step (int): The time-step of the intervention.
        num_cf_samples (int, optional): Number of posterior samples to use for counterfactual inference. Defaults to 100.
        outcome_target (Optional[float], optional): Optional outcome target. If given, the effect is computed for that specific outcome. Defaults to None.
        seed (Optional[int], optional): Optional seed used for reproducibility. Defaults to None.
    Returns:
        int: The total counterfactual effect of the intervention.
    """
    assert 1 == len(intervention.keys()), "TCFE only supports single interventions."
    total = 0

    for i in range(num_cf_samples):
        rng = np.random.default_rng(seed + i)

        # sample noise from posterior
        act_noise, env_noise = env.sample_noise_from_posterior(trajectory=trajectory, agents=agents, rng=rng)

        # construct the intervention set
        do_operators = [{} for _ in range(trajectory.horizon)]
        do_operators[time_step] = intervention

        # sample counterfactual trajectory for tcfe
        traj_tcfe = env.sample_trajectory(agents, act_noise=act_noise, env_noise=env_noise, do_operators=do_operators,
                                        initial_state=trajectory.states[0], horizon=trajectory.horizon, rng=rng)

        # update total
        if outcome_target is not None:
            total += isclose(traj_tcfe.outcome(), outcome_target, rel_tol=1e-05, abs_tol=0.0)
        else:
            total += traj_tcfe.outcome()

    # compute value for reference
    if outcome_target is not None:
        ref = isclose(trajectory.outcome(), outcome_target, rel_tol=1e-05, abs_tol=0.0)
    else:
        ref = trajectory.outcome()

    return total / num_cf_samples - ref


def pse(trajectory: Trajectory, env: CausalEnv, agents: List[CausalActor], intervention: Dict[int, int], time_step: int, num_cf_samples: int = 100, outcome_target: Optional[float] = None, seed: Optional[int] = None) -> float:
    """Implements the path-specific effect baseline.
    Args:
        trajectory (Trajectory): The trajectory to calculate the path-specific effect for.
        env (CausalEnv): Instance of the environment to sample counterfactuals from.
        agents (List[CausalActor]): List of agents acting in the environment.
        intervention (Dict[int, int]): The intervention to apply, with keys indicating agent's id and values indicating agent's action.
        time_step (int): The time-step of the intervention.
        num_cf_samples (int, optional): Number of posterior samples to use for counterfactual inference. Defaults to 100.
        outcome_target (Optional[float], optional): Optional outcome target. If given, the effect is computed for that specific outcome. Defaults to None.
        seed (Optional[int], optional): Optional seed used for reproducibility. Defaults to None.
    Returns:
        int: The path-specific effect of the intervention.
    """
    assert 1 == len(intervention.keys()), "PSE only supports single interventions."
    total = 0

    for i in range(num_cf_samples):
        rng = np.random.default_rng(seed + i)

        # sample noise from posterior
        act_noise, env_noise = env.sample_noise_from_posterior(trajectory=trajectory, agents=agents, rng=rng)

        # construct the intervention
        do_agents = list(intervention.keys())
        do_operators = [{} for _ in range(trajectory.horizon)]
        do_operators[time_step] = intervention
        for t in range(time_step + 1, trajectory.horizon):
            do_operators[t] = {agent_id: trajectory.actions[t][agent_id] for agent_id in do_agents}

        # sample counterfactual trajectory for pse
        traj_pse = env.sample_trajectory(agents, act_noise=act_noise, env_noise=env_noise, do_operators=do_operators,
                                        rng=rng, initial_state=trajectory.states[0], horizon=trajectory.horizon)

        # update total
        if outcome_target is not None:
            total += isclose(traj_pse.outcome(), outcome_target, rel_tol=1e-05, abs_tol=0.0)
        else:
            total += traj_pse.outcome()

    # compute value for reference
    if outcome_target is not None:
        ref = isclose(trajectory.outcome(), outcome_target, rel_tol=1e-05, abs_tol=0.0)
    else:
        ref = trajectory.outcome()

    return total / num_cf_samples - ref


def ase(trajectory: Trajectory, env: CausalEnv, agents: List[CausalActor], intervention: Dict[int, int], effect_agents: List[int], time_step: int, num_cf_samples: int = 100, outcome_target: Optional[float] = None, seed: Optional[int] = None) -> float:
    """Implements the agent-specific effect.
    Args:
        trajectory (Trajectory): The trajectory to calculate the agent-specific effect for.
        env (CausalEnv): Instance of the environment to sample counterfactuals from.
        agents (List[CausalActor]): List of agents acting in the environment.
        intervention (Dict[int, int]): The intervention to apply, with keys indicating agent's id and values indicating agent's action.
        effect_agents: (List[int]): List of effect agent ids.
        time_step (int): The time-step of the intervention.
        num_cf_samples (int, optional): Number of posterior samples to use for counterfactual inference. Defaults to 100.
        outcome_target (Optional[float], optional): Optional outcome target. If given, the effect is computed for that specific outcome. Defaults to None.
        seed (Optional[int], optional): Optional seed used for reproducibility. Defaults to None.
    Returns:
        int: The agent-specific effect of the intervention.
    """
    assert 1 == len(intervention.keys()), "ASE only supports single interventions."
    total = 0

    for i in range(num_cf_samples):
        rng = np.random.default_rng(seed + i)

        # sample noise from posterior
        act_noise, env_noise = env.sample_noise_from_posterior(trajectory=trajectory, agents=agents, rng=rng)

        # sample counterfactual trajectory
        do_operators = [{} for _ in range(trajectory.horizon)]
        do_operators[time_step] = intervention
        traj_cf = env.sample_trajectory(agents, act_noise=act_noise, env_noise=env_noise, do_operators=do_operators,
                                        initial_state=trajectory.states[0], rng=rng, pad=True, horizon=trajectory.horizon)

        # construct the intervention set
        do_operators = [{} for _ in range(trajectory.horizon)]
        non_effect_agents = [agent.id for agent in agents if agent.id not in effect_agents]

        for t in range(time_step + 1, trajectory.horizon):
            for agent_id in non_effect_agents:
                do_operators[t][agent_id] = trajectory.actions[t][agent_id]
            for agent_id in effect_agents:
                do_operators[t][agent_id] = traj_cf.actions[t][agent_id]

        # sample counterfactual trajectory for ase
        traj_ase = env.sample_trajectory(agents, act_noise=act_noise, env_noise=env_noise, do_operators=do_operators,
                                        initial_state=trajectory.states[0], rng=rng, horizon=trajectory.horizon)

        # update total
        if outcome_target is not None:
            total += isclose(traj_ase.outcome(), outcome_target, rel_tol=1e-05, abs_tol=0.0)
        else:
            total += traj_ase.outcome()

    # compute value for reference
    if outcome_target is not None:
        ref = isclose(trajectory.outcome(), outcome_target, rel_tol=1e-05, abs_tol=0.0)
    else:
        ref = trajectory.outcome()

    return total / num_cf_samples - ref


def ase_sv(trajectory: Trajectory, env: CausalEnv, agents: List[CausalActor], intervention: Dict[int, int], time_step: int, num_cf_samples: int = 100, outcome_target: Optional[float] = None, seed: Optional[int] = None) -> dict:
    """Implements the agent-specific effect Shapley value method.
    Args:
        trajectory (Trajectory): The trajectory to calculate ASE-SV for.
        env (CausalEnv): Instance of the environment to sample counterfactuals from.
        agents (List[CausalActor]): List of agents acting in the environment.
        intervention (Dict[int, int]): The intervention to apply, with keys indicating agent's id and values indicating agent's action.
        time_step (int): The time-step of the intervention.
        num_cf_samples (int, optional): Number of posterior samples to use for counterfactual inference. Defaults to 100.
        outcome_target (Optional[float], optional): Optional outcome target. If given, the effect is computed for that specific outcome. Defaults to None.
        seed (Optional[int], optional): Optional seed used for reproducibility. Defaults to None.
    Returns:
        dict: The ASE-SV for all agents.
    """
    agent_ids = [agent.id for agent in agents]
    # form all coalitions
    coalitions = list(itertools.chain.from_iterable(itertools.combinations(agent_ids, l) for l in range(0, len(agent_ids) + 1)))
    coalitions = list(map(list, coalitions))
    sv = {}

    # assign a contribution score to each agent by measuring its ASE-SV
    for i in agent_ids:
        weights, marg_contrs = [], []
        for S in coalitions:
            if i in S:
                continue
            # calculate coefficient
            w_s = factorial(len(S)) * factorial(len(agents) - len(S) - 1) / factorial(len(agents))
            # calculate marginal contribution
            ase_with_i = ase(trajectory, env, agents, intervention, effect_agents=S + [i], time_step=time_step, num_cf_samples=num_cf_samples, outcome_target=outcome_target, seed=seed)
            ase_wo_i = ase(trajectory, env, agents, intervention, effect_agents=S, time_step=time_step, num_cf_samples=num_cf_samples, outcome_target=outcome_target, seed=seed)
            marg_contr = ase_with_i - ase_wo_i

            weights.append(w_s)
            marg_contrs.append(marg_contr)

        sv[i] = sum([w * d for w, d in zip(weights, marg_contrs)])

    total_ase = ase(trajectory, env, agents, intervention, [a.id for a in agents], time_step, num_cf_samples, outcome_target, seed)
    assert isclose(sum(sv.values()), total_ase, rel_tol=1e-05, abs_tol=0.0), "Decomposition of total ASE failed."

    return sv


def sse(trajectory: Trajectory, env: CausalEnv, agents: List[CausalActor], intervention: Dict[int, int], time_step: int, num_cf_samples: int = 100, outcome_target: Optional[float] = None, seed: Optional[int] = None) -> float:
    """Implements the state-specific effect.
    Args:
        trajectory (Trajectory): The trajectory to calculate the state-specific effect for.
        env (CausalEnv): Instance of the environment to sample counterfactuals from.
        agents (List[CausalActor]): List of agents acting in the environment.
        intervention (Dict[int, int]): The intervention to apply, with keys indicating agent's id and values indicating agent's action.
        time_step (int): The time-step of the intervention.
        num_cf_samples (int, optional): Number of posterior samples to use for counterfactual inference. Defaults to 100.
        outcome_target (Optional[float], optional): Optional outcome target. If given, the effect is computed for that specific outcome. Defaults to None.
        seed (Optional[int], optional): Optional seed used for reproducibility. Defaults to None.
    Returns:
        int: The state-specific effect of the intervention.
    """
    assert 1 == len(intervention.keys()), "SSE only supports single interventions."
    total = 0

    for i in range(num_cf_samples):
        rng = np.random.default_rng(seed + i)

        # sample noise from posterior
        act_noise, env_noise = env.sample_noise_from_posterior(trajectory=trajectory, agents=agents, rng=rng)

        # construct the intervention set
        do_operators = [{} for _ in range(trajectory.horizon)]
        do_operators[time_step] = intervention
        for t in range(time_step + 1, trajectory.horizon):
            # TODO
            for agent in agents:
                if len(agents) == 1:
                    do_operators[t][agent.id] = trajectory.actions[t]
                else:
                    do_operators[t][agent.id] = trajectory.actions[t][agent.id]

        # sample counterfactual trajectory for sse
        traj_sse = env.sample_trajectory(agents, act_noise=act_noise, env_noise=env_noise, do_operators=do_operators,
                                        rng=rng, initial_state=trajectory.states[0], horizon=trajectory.horizon)

        # update total
        if outcome_target is not None:
            total += isclose(traj_sse.outcome(), outcome_target, rel_tol=1e-05, abs_tol=0.0)
        else:
            total += traj_sse.outcome()

    # compute value for reference
    if outcome_target is not None:
        ref = isclose(trajectory.outcome(), outcome_target, rel_tol=1e-05, abs_tol=0.0)
    else:
        ref = trajectory.outcome()

    return total / num_cf_samples - ref


def sse_variance(trajectory: Trajectory, env: CausalEnv, agents: List[CausalActor], intervention: Dict[int, int], time_step: int, var_time_step: int, num_cf_samples_cond: int = 20, num_cf_samples: int = 100, seed: Optional[int] = None) -> float:
    """Implements the (noise-conditional) variance measure for sse.
    Args:
        trajectory (Trajectory): The trajectory to calculate SSE variance for.
        env (CausalEnv): Instance of the environment to sample counterfactuals from.
        agents (List[CausalActor]): List of agents acting in the environment.
        intervention (Dict[int, int]): The intervention to apply, with keys indicating agent's id and values indicating agent's action.
        time_step (int): The time-step of the intervention.
        var_time_step (int): We measure the variance conditioned on all noise variables that come chronologically before var_time_step.
        num_cf_samples_cond (int, optional): Number of posterior samples to use for counterfactual inference of conditioning variables. Defaults to 20.
        num_cf_samples (int, optional): Number of posterior samples to use for counterfactual inference of non-conditioning variables. Defaults to 100.
        seed (Optional[int], optional): Optional seed used for reproducibility. Defaults to None.
    Returns:
        int: The SSE variance.
    """
    assert 1 == len(intervention.keys()), "SSE variance computation only supports single interventions."

    rng = np.random.default_rng(seed)
    mean1, mean2 = 0, 0

    for _ in range(num_cf_samples_cond):
        # sample conditioning noise from posterior
        act_noise_cond, env_noise_cond = env.sample_noise_from_posterior(trajectory=trajectory, agents=agents, rng=rng)

        c1, c2 = 0, 0
        for _ in range(num_cf_samples):
            # sample non-conditioning noise from posterior
            act_noise_non, env_noise_non = env.sample_noise_from_posterior(trajectory=trajectory, agents=agents, rng=rng)

            # fix conditioning noise
            act_noise = act_noise_cond[:var_time_step] + act_noise_non[var_time_step:]
            env_noise = env_noise_cond[:var_time_step] + env_noise_non[var_time_step:]

            # construct the intervention set
            do_operators = [{} for _ in range(trajectory.horizon)]
            do_operators[time_step] = intervention
            for t in range(time_step + 1, trajectory.horizon):
                for agent in agents:
                    if len(agents) == 1:
                        do_operators[t][agent.id] = trajectory.actions[t]
                    else:
                        do_operators[t][agent.id] = trajectory.actions[t][agent.id]

            # sample counterfactual trajectory for sse
            traj_sse = env.sample_trajectory(agents, act_noise=act_noise, env_noise=env_noise, do_operators=do_operators,
                                            rng=rng, initial_state=trajectory.states[0], horizon=trajectory.horizon)

            # update c1 and c2 -- Var(X - a) = Var(X)
            c = traj_sse.outcome()
            c1 += c
            c2 += c ** 2

        mean1 += (c1 / num_cf_samples) ** 2
        mean2 += c2 / num_cf_samples

    return (mean2 - mean1) / num_cf_samples_cond


def sse_icc(trajectory: Trajectory, env: CausalEnv, agents: List[CausalActor], intervention: Dict[int, int], time_step: int, num_cf_samples_cond: int = 20, num_cf_samples: int = 100, outcome_target: Optional[float] = None, min_var: Optional[float] = 1e-06, sse_value: Optional[float] = None, seed: Optional[int] = None) -> list:
    """Implements the state-specific effect intrinsic causal contribution method.
    Args:
        trajectory (Trajectory): The trajectory to calculate SSE-ICC for.
        env (CausalEnv): Instance of the environment to sample counterfactuals from.
        agents (List[CausalActor]): List of agents acting in the environment.
        intervention (Dict[int, int]): The intervention to apply, with keys indicating agent's id and values indicating agent's action.
        time_step (int): The time-step of the intervention.
        num_cf_samples_cond (int, optional): Number of posterior samples to use for counterfactual inference of conditioning variables. Defaults to 20.
        num_cf_samples (int, optional): Number of posterior samples to use for counterfactual inference of non-conditioning variables. Defaults to 100.
        outcome_target (Optional[float], optional): Optional outcome target. If given, the effect is computed for that specific outcome. Defaults to None.
        min_var (Optional[float], optional): Optional: Minimum variance that is considered for assinging positive contribution. Defaults to 1e-06.
        sse_value (Optional[float], optional): Optional (pre-computed) SSE value. Defaults to None.
        seed (Optional[int], optional): Optional seed used for reproducibility. Defaults to None.
    Returns:
        list: The SSE-ICC for all state variables.
    """
    # compute SSE if not given
    if sse_value is None:
        sse_value = sse(trajectory, env, agents, intervention, time_step, num_cf_samples, outcome_target, seed)
    # initialize contribution scores
    h = [0] * trajectory.horizon
    # if sse is zero, assign zero contribution scores to all state variables
    if sse_value == 0:
        return h
    # compute total variance
    total_var = sse_variance(trajectory, env, agents, intervention, time_step, var_time_step=0,
                             num_cf_samples_cond=num_cf_samples_cond, num_cf_samples=num_cf_samples, seed=seed)
    # if total variance is close to zero, assign zero contribution scores to all state variables
    if total_var < min_var:
        return h

    # assign a contribution score to each state variable by measuring its marginal ICC to the effect
    var_prev = total_var
    for t in range(time_step + 1, trajectory.horizon):
        # measure variance when (additionally) conditioning on variables of time-step t
        var_cond_t = sse_variance(trajectory, env, agents, intervention, time_step, var_time_step=t+1,
                     num_cf_samples_cond=num_cf_samples_cond, num_cf_samples=num_cf_samples, seed=seed)
        assert var_cond_t >= 0, "Variance cannot be negative."
        # compute marginal ICC
        icc = var_prev - var_cond_t
        # compute contribution score
        h[t] = (icc / total_var) * sse_value
        # if remaining variance is close to zero, then assing zero contribution scores to all future state variables
        if var_cond_t < min_var:
            break
        var_prev = var_cond_t

    return h


def reverse_sse(trajectory: Trajectory, env: CausalEnv, agents: List[CausalActor], intervention: Dict[int, int], time_step: int, num_cf_samples: int = 100, outcome_target: Optional[float] = None, seed: Optional[int] = None) -> float:
    """Implements the reverse state-specific effect.
    Args:
        trajectory (Trajectory): The trajectory to calculate the reverse state-specific effect for.
        env (CausalEnv): Instance of the environment to sample counterfactuals from.
        agents (List[CausalActor]): List of agents acting in the environment.
        intervention (Dict[int, int]): The intervention to apply, with keys indicating agent's id and values indicating agent's action.
        time_step (int): The time-step of the intervention.
        num_cf_samples (int, optional): Number of posterior samples to use for counterfactual inference. Defaults to 100.
        outcome_target (Optional[float], optional): Optional outcome target. If given, the effect is computed for that specific outcome. Defaults to None.
        seed (Optional[int], optional): Optional seed used for reproducibility. Defaults to None.
    Returns:
        int: The reverse state-specific effect of the intervention.
    """
    assert 1 == len(intervention.keys()), "reverse SSE only supports single interventions."
    total = 0
    total_cf = 0

    for i in range(num_cf_samples):
        rng = np.random.default_rng(seed + i)

        # sample noise from posterior
        act_noise, env_noise = env.sample_noise_from_posterior(trajectory=trajectory, agents=agents, rng=rng)

        # sample counterfactual trajectory
        do_operators = [{} for _ in range(trajectory.horizon)]
        do_operators[time_step] = intervention
        traj_cf = env.sample_trajectory(agents, act_noise=act_noise, env_noise=env_noise, do_operators=do_operators,
                                        initial_state=trajectory.states[0], rng=rng, pad=True, horizon=trajectory.horizon)

        # update total_cf
        if outcome_target is not None:
            total_cf += isclose(traj_cf.outcome(), outcome_target, rel_tol=1e-05, abs_tol=0.0)
        else:
            total_cf += traj_cf.outcome()

        # construct the intervention set
        do_operators = [{} for _ in range(trajectory.horizon)]
        for t in range(time_step + 1, trajectory.horizon):
            for agent in agents:
                if len(agents) == 1:
                    do_operators[t][agent.id] = traj_cf.actions[t]
                else:
                    do_operators[t][agent.id] = traj_cf.actions[t][agent.id]

        # sample counterfactual trajectory for reverse_sse
        traj_rsse = env.sample_trajectory(agents, act_noise=act_noise, env_noise=env_noise, do_operators=do_operators,
                                        rng=rng, initial_state=trajectory.states[0], horizon=trajectory.horizon)

        # update total
        if outcome_target is not None:
            total += isclose(traj_rsse.outcome(), outcome_target, rel_tol=1e-05, abs_tol=0.0)
        else:
            total += traj_rsse.outcome()

    return (total - total_cf) / num_cf_samples


def reverse_sse_variance(trajectory: Trajectory, env: CausalEnv, agents: List[CausalActor], intervention: Dict[int, int], time_step: int, var_time_step: int, num_cf_samples_cond: int = 20, num_cf_samples: int = 100, seed: Optional[int] = None) -> float:
    """Implements the (noise-conditional) variance measure for reverse sse.
    Args:
        trajectory (Trajectory): The trajectory to calculate reverse SSE variance for.
        env (CausalEnv): Instance of the environment to sample counterfactuals from.
        agents (List[CausalActor]): List of agents acting in the environment.
        intervention (Dict[int, int]): The intervention to apply, with keys indicating agent's id and values indicating agent's action.
        time_step (int): The time-step of the intervention.
        var_time_step (int): We measure the variance conditioned on all noise variables that come chronologically before var_time_step.
        num_cf_samples_cond (int, optional): Number of posterior samples to use for counterfactual inference of conditioning variables. Defaults to 20.
        num_cf_samples (int, optional): Number of posterior samples to use for counterfactual inference of non-conditioning variables. Defaults to 100.
        seed (Optional[int], optional): Optional seed used for reproducibility. Defaults to None.
    Returns:
        int: The reverse SSE variance.
    """
    assert 1 == len(intervention.keys()), "reverse SSE variance computation only supports single interventions."

    rng = np.random.default_rng(seed)
    mean1, mean2 = 0, 0

    for _ in tqdm.tqdm(range(num_cf_samples_cond)):
        # sample conditioning noise from posterior
        act_noise_cond, env_noise_cond = env.sample_noise_from_posterior(trajectory=trajectory, agents=agents, rng=rng)

        c1, c2 = 0, 0
        for _ in range(num_cf_samples):
            # sample non-conditioning noise from posterior
            act_noise_non, env_noise_non = env.sample_noise_from_posterior(trajectory=trajectory, agents=agents, rng=rng)

            # fix conditioning noise
            act_noise = act_noise_cond[:var_time_step] + act_noise_non[var_time_step:]
            env_noise = env_noise_cond[:var_time_step] + env_noise_non[var_time_step:]

            # sample counterfactual trajectory
            do_operators = [{} for _ in range(trajectory.horizon)]
            do_operators[time_step] = intervention
            traj_cf = env.sample_trajectory(agents, act_noise=act_noise, env_noise=env_noise, do_operators=do_operators,
                                            initial_state=trajectory.states[0], rng=rng, pad=True, horizon=trajectory.horizon)

            # construct the intervention set
            do_operators = [{} for _ in range(trajectory.horizon)]
            for t in range(time_step + 1, trajectory.horizon):
                for agent in agents:
                    if len(agents) == 1:
                        do_operators[t][agent.id] = traj_cf.actions[t]
                    else:
                        do_operators[t][agent.id] = traj_cf.actions[t][agent.id]

            # sample counterfactual trajectory for reverse_sse
            traj_rsse = env.sample_trajectory(agents, act_noise=act_noise, env_noise=env_noise, do_operators=do_operators,
                                            rng=rng, initial_state=trajectory.states[0], horizon=trajectory.horizon)

            # update c1 and c2
            c = traj_rsse.outcome() - traj_cf.outcome()
            c1 += c
            c2 += c ** 2

        mean1 += (c1 / num_cf_samples) ** 2
        mean2 += c2 / num_cf_samples

    return (mean2 - mean1) / num_cf_samples_cond


def reverse_sse_icc(trajectory: Trajectory, env: CausalEnv, agents: List[CausalActor], intervention: Dict[int, int], time_step: int, num_cf_samples_cond: int = 20, num_cf_samples: int = 100, outcome_target: Optional[float] = None, min_var: Optional[float] = 1e-06, reverse_sse_value: Optional[float] = None, seed: Optional[int] = None) -> list:
    """Implements the reverse state-specific effect intrinsic causal contribution method.
    Args:
        trajectory (Trajectory): The trajectory to calculate reverse SSE-ICC for.
        env (CausalEnv): Instance of the environment to sample counterfactuals from.
        agents (List[CausalActor]): List of agents acting in the environment.
        intervention (Dict[int, int]): The intervention to apply, with keys indicating agent's id and values indicating agent's action.
        time_step (int): The time-step of the intervention.
        num_cf_samples_cond (int, optional): Number of posterior samples to use for counterfactual inference of conditioning variables. Defaults to 20.
        num_cf_samples (int, optional): Number of posterior samples to use for counterfactual inference of non-conditioning variables. Defaults to 100.
        outcome_target (Optional[float], optional): Optional outcome target. If given, the effect is computed for that specific outcome. Defaults to None.
        min_var (Optional[float], optional): Optional: Minimum variance that is considered for assinging positive contribution. Defaults to 1e-06.
        reverse_sse_value: Optional (pre-computed) reverse SSE value. Defaults to None.
        seed (Optional[int], optional): Optional seed used for reproducibility. Defaults to None.
    Returns:
        list: The reverse SSE-ICC for all state variables.
    """
    # compute reverse SSE if not given
    if reverse_sse_value is None:
        reverse_sse_value = reverse_sse(trajectory, env, agents, intervention, time_step, num_cf_samples, outcome_target, seed)
    # initialize contribution scores
    h = [0] * trajectory.horizon
    # if reverse sse is zero, assign zero contribution scores to all state variables
    if reverse_sse_value == 0:
        return h
    # compute total variance
    total_var = reverse_sse_variance(trajectory, env, agents, intervention, time_step, var_time_step=0,
                             num_cf_samples_cond=num_cf_samples_cond, num_cf_samples=num_cf_samples, seed=seed)
    # if total variance is close to zero, assign zero contribution scores to all state variables
    if total_var < min_var:
        return h

    # assign a contribution score to each state variable by measuring its marginal ICC to the effect
    var_prev = total_var
    for t in range(time_step + 1, trajectory.horizon):
        # measure variance when (additionally) conditioning on variables of time-step t
        var_cond_t = reverse_sse_variance(trajectory, env, agents, intervention, time_step, var_time_step=t+1,
                     num_cf_samples_cond=num_cf_samples_cond, num_cf_samples=num_cf_samples, seed=seed)
        assert var_cond_t >= 0 or isclose(abs(var_cond_t), 0.0, rel_tol=1e-05, abs_tol=1e-05), f"Variance is always non-negative: {var_cond_t}."
        # compute marginal ICC
        icc = var_prev - var_cond_t
        # compute contribution score
        h[t] = (icc / total_var) * reverse_sse_value
        # if remaining variance is close to zero, then assing zero contribution scores to all future state variables
        if var_cond_t < min_var:
            break
        var_prev = var_cond_t

    return h
