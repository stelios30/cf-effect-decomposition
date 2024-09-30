import numpy as np

from numpy.random import Generator
from typing import Tuple, Optional, List

from ced.tools.order import Order
from ced.tools.utils import find_range_for_item, get_probability_ranges

from .order import Order


class NoiseModel:
    """Describes an environment noise model."""

    def __init__(self, order: Optional[Order] = None) -> None:
        self.order = order

    def sample(self, rng: Optional[Generator] = None) -> "NoiseModel":
        """Samples a new noise model instance."""
        raise NotImplementedError()

    def sample_posterior(self, probs: List[float], realised: int, rng: Optional[Generator] = None) -> "NoiseModel":
        """Samples a new noise model that corresponds to a realised action. The `realised` element is always
           considered to be sampled from the `probs` distribution (i.e., it represents an index of the element
           in the `probs` distribution).
        """
        raise NotImplementedError()

    def choice(self, probs: List[float], rng: Optional[Generator] = None) -> int:
        """Samples an index of element from the passed categorical distribution `probs`, taking noise into
           consideration. If order is specified, it returns the index of the sampled element w.r.t. the
           passed `probs` distribution ordered by `self.order`.
        """
        raise NotImplementedError()


class NullNoiseModel(NoiseModel):
    """Implements the default "no-noise" model."""

    def sample(self, rng: Optional[Generator] = None) -> "NoiseModel":
        return NullNoiseModel()

    def sample_posterior(self, probs: List[float], realised: int, rng: Optional[Generator] = None) -> "NoiseModel":
        return NullNoiseModel()

    def choice(self, probs: List[float], rng: Optional[Generator] = None) -> int:
        rng = rng or np.random.default_rng()
        return rng.choice(list(range(len(probs))), p=probs)


class UniformNoiseModel(NoiseModel):
    """Implements the uniform noise model.
       If the order is given, then this model implements noise monotonicity.
    """

    def __init__(self, noise: Optional[float] = None, order: Optional[Order] = None) -> None:
        super().__init__(order=order)
        self.noise = noise

    def sample(self, rng: Optional[Generator] = None) -> "NoiseModel":
        rng = rng or np.random.default_rng()
        noise = rng.uniform(low=0, high=1, size=1).item()
        return UniformNoiseModel(noise, self.order)

    def sample_posterior(self, probs: List[float], realised: int, rng: Optional[Generator] = None) -> "NoiseModel":
        if self.order:
            # applies total order to input probabilities
            index = self.order.sort(list(range(len(probs))), return_index=True)
            probs = [probs[ind] for ind in index]
            realised = index.index(realised)

        rng = rng or np.random.default_rng()
        ranges = get_probability_ranges(probs)
        noise = rng.uniform(low=ranges[realised][0], high=ranges[realised][1], size=1).item()
        return UniformNoiseModel(noise, self.order)

    def choice(self, probs: List[float], rng: Optional[Generator] = None) -> int:
        assert self.noise is not None, "Noise must be sampled, before calling `choice`."

        if self.order:
            # applies total order to the generated ranges
            index = self.order.sort(list(range(len(probs))), return_index=True)
            probs = [probs[ind] for ind in index]
        else:
            index = None

        ranges = get_probability_ranges(probs)
        item = [r[0] <= self.noise and self.noise < r[1] for r in ranges]
        item = np.array(item).astype(int).argmax()

        return index[item] if index is not None else item


class SepsisStateUniformNoiseModel(UniformNoiseModel):
    """Implements the uniform noise model.
       This model is identical in function to "UniformNoiseModel", but it implements a performance
       optimization specific to the sepsis environment. In more details, instead of dynamically
       constructing probability ranges every time a `choice` method is called, it expects a
       pre-computed probability ranges to be passed (see also module `ced.envs.sepsis`).
    """

    def sample(self, rng: Optional[Generator] = None) -> "SepsisStateUniformNoiseModel":
        noise = super().sample(rng)
        return SepsisStateUniformNoiseModel(noise.noise, self.order)

    def sample_posterior(self, probs: List[float], realised: int, rng: Optional[Generator] = None) -> "SepsisStateUniformNoiseModel":
        rng = rng or np.random.default_rng()
        ranges = get_probability_ranges(probs)
        noise = rng.uniform(low=ranges[realised][0], high=ranges[realised][1], size=1).item()
        return SepsisStateUniformNoiseModel(noise, self.order)

    def choice(self, ranges: List[Tuple[float]]) -> int:
        assert self.noise is not None, "Noise must be sampled, before calling `choice`."

        index = self.order.decode_index if self.order else None
        item = find_range_for_item(item=self.noise, ranges=ranges)
        assert item != -1, f"Item {self.noise} not found in ranges {ranges}."

        return index[item] if index is not None else item


NOISE_MODELS = {
    "null": NullNoiseModel,
    "monotonic": UniformNoiseModel,
}
