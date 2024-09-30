from typing import List, Literal


class Order:
    """Implements a total order of a list of items."""

    def __init__(self, order: List[int]) -> None:
        self.order = order                          # for a given order e.g. [1, 2, 0]
        self.ground_truth = list(range(len(order))) # assumes a default order of [0, 1, 2]
        self.encode_items = {item: self.order.index(item) for item in self.ground_truth}
        self.encode_index = [self.order.index(item) for item in self.ground_truth]
        self.decode_index = [self.ground_truth.index(item) for item in self.order]

    def __str__(self) -> str:
        return ", ".join([str(item) for item in self.order])

    def compare(self, a: int, b: int) -> Literal[-1, 0, 1]:
        a_ind = self.encode_items[a]
        b_ind = self.encode_items[b]
        return -1 if a_ind < b_ind else 1 if a_ind > b_ind else 0

    def sort(self, items: List[int], return_index: bool = False) -> List[int]:
        if return_index:
            return self.decode_index
        return [items[self.encode_index[i]] for i in range(len(items))]
