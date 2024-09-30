import torch
import sys
from math import isclose
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numpy.random import Generator
from os.path import join
from tianshou.policy import BasePolicy

if TYPE_CHECKING:
    from ced.envs.causal import CausalActor, CausalEnv


def sample_trajectories(env: "CausalEnv", agents: List["CausalActor"], num_trajectories: int, outcome_target: Optional[float] = None, rng: Optional[Generator] = None):
    """Samples trajectories from the passed environment.
    Args:
        env (CausalEnv): Environment to sample trajectories from.
        agents (List[&quot;CausalActor&quot;]): List of agents acting in the environment.
        num_trajectories (int): Number of trajectories to sample.
        outcome_target (Optional[float], optional): Optional outcome target. If given, only trajectories with a given outcome are sampled.
        rng (Optional[Generator], optional): Fixed random number generator, used for reproducibility. Defaults to None.
    Returns:
        List[Trajectory]: List of sampled trajectories.
    """
    trajectories = []

    while len(trajectories) < num_trajectories:
        trajectory = env.sample_trajectory(agents, rng=rng)
        trajectory.id = len(trajectories)

        if outcome_target is not None and isclose(trajectory.outcome(), outcome_target):
            trajectories.append(trajectory)
        elif outcome_target is None:
            trajectories.append(trajectory)

    return trajectories


def find_range_for_item(item: float, ranges: List[Tuple[float]]) -> int:
    """Finds a range that contains an item using binary search.
    Args:
        item (float): Item to find.
        ranges (List[Tuple[float]]): List of ranges, sorted in ascending order (e.g., [(0.0, 0.25), (0.25, 0.75), (0.75, 1.0)]).
    Returns:
        int: Index of the range that contains the item
    """
    def _bin_search(low_ind: int, high_ind: int) -> int:
        if low_ind > high_ind:
            return -1

        mid_ind = (high_ind + low_ind) // 2

        if ranges[mid_ind][0] <= item and item < ranges[mid_ind][1]:
            return mid_ind
        elif item < ranges[mid_ind][0]:
            return _bin_search(low_ind, mid_ind - 1)
        else:
            return _bin_search(mid_ind + 1, high_ind)

    return _bin_search(0, len(ranges) - 1)


def get_probability_ranges(probs: List[float]) -> List[Tuple[float]]:
    """Constructs a list of ranges from a probability distribution.
    Args:
        probs (List[float]): Probability distribution (e.g., [0.25, 0.25, 0.50]).
    Returns:
        List[Tuple[float]]: List of probability ranges (e.g., [(0.0, 0.25), (0.25, 0.50), (0.50, 1.0)])
    """
    result, limit = [], 0.0
    for prob in probs:
        result.append((limit, limit + prob))
        limit += prob
    return result


def find_by_id(items: List[Any], id: Any) -> Optional[Any]:
    """Helper function that finds an item in a list by its id.
    Args:
        items (List[Any]): List of items to search.
        id (Any): Item's id to search for.
    Returns:
        Optional[Any]: Item with the given id or None if not found.
    """
    for item in items:
        if hasattr(item, "id") and item.id == id:
            return item
    return None


def export_legend(figure: mpl.figure.Figure, axes: mpl.axes.Axes):
    """Exports a legend from a figure."""
    handles, labels = axes.legend_.legendHandles, [t.get_text() for t in axes.legend_.get_texts()]
    legend = axes.get_legend()
    legend_bbox = legend.get_tightbbox(figure.canvas.get_renderer())
    legend_bbox = legend_bbox.transformed(figure.dpi_scale_trans.inverted())
    legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))
    legend_ax.axis("off")
    legend_squared = legend_ax.legend(
        handles, labels,
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=legend_fig.transFigure,
        frameon=True,
        facecolor="#ffffff",
        fancybox=True,
        shadow=False,
        ncol=min(len(labels), 3),
        fontsize=20,
        title=legend.get_title().get_text(),
    )
    return legend_fig, legend_squared


def decay_schedule(init_value: float, min_value: float = 0.0, max_steps: int = 1.0, decay_ratio: float = 0.9):
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps
    values = np.logspace(-2, 0, decay_steps, endpoint=True)
    values = values[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), "edge")
    return values


def configure_logger(log_dir: Optional[str] = None, log_epochs: bool = True):
    logger.remove()

    log_format = "<green>[{time:DD.MM.YYYY at HH:mm:ss}]</green>"
    if log_epochs:
        log_format = " EPOCH={extra[epoch]} " + log_format
    log_format = log_format + " {message}"

    logger.add(sys.stdout, colorize=True, format=log_format, backtrace=True, diagnose=True)
    if log_dir:
        logger.add(join(log_dir, "out.log"), format=log_format, backtrace=True, diagnose=True)

    return logger


def to_device(policy: BasePolicy, device: str) -> BasePolicy:
    """Helper function to transfer an instance of a Tianshou policy to the target device.
    Args:
        policy (BasePolicy): Instance of the Tianshou policy.
        device (str): Target PyTorch device.
    """
    result = policy.to(device)
    policy = result

    while True:
        if not hasattr(policy, "model") or not hasattr(policy.model, "device"):
            return result
        policy.model.device = device
        policy = policy.model
