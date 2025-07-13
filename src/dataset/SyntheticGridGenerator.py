"""
Synthetic grid dataset generator. Used for generating the data used for training the neural network
"""

from typing import Optional


class SyntheticGridGenerator:
    def __init__(self, width: int, height: int, algorithm: str):
        self._width = width
        self._height = height
        self._algorithm = algorithm
        self.grid: Optional[list[list[int]]] = None

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        allowed = ("dfs", "prim", "kruskal")
        if value not in allowed:
            raise ValueError(
                f"[SyntheticGridGenerator] Algorithm {value} is not supported. Choose from [{allowed}]"
            )
        self._algorithm = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if 1 < value > 10:
            raise ValueError(
                f"[SyntheticGridGenerator] Width {value} is not supported; must be 1 <= width <= 10"
            )
        self._height = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        if 1 < value > 10:
            raise ValueError(
                f"[SyntheticGridGenerator] Height {value} is not supported; must be 1 <= height <= 10"
            )
        self._height = value

    def generate_grid(self) -> int:
        match self._algorithm:
            case "dfs":
                print("dfs")
            case "prim":
                print("prim")
            case "kruskal":
                print("kruskal")
        return 0
