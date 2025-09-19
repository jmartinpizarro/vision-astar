"""
This file contains the State definition used for the
IDA* algorithm used for the pathfinding
"""

from typing import List, Tuple


# State class definition
class State:
    def __init__(self, size: int, matrix: List[List[str]], x: int, y: int):
        """
        Definition of an instance of an State.

        @param size: size -> of the squared grid
        @param matrix -> the input matrix.
        @param x: int -> position of the player on the x-axis.
        @param y: int -> position of the player on the y-axis.
        """
        self.size = size
        self.matrix = matrix

        self.x = x
        self.y = y

    # Methods
    def children(self, goalState) -> List[Tuple[int, "State"]]:
        """
        Returns a sorted array with all possible children in a tuple (heuristic, State)

        param goalState: State -> the goal state the alg is looking for
        @returns a sorted array with all the states
        """
        sucessors = []

        # it is not allowed octile movements
        # to the left
        if self.x > 0:
            if self.matrix[self.x - 1][self.y] != "X":
                sucessorNode = State(self.size, self.matrix, self.x - 1, self.y)
                h = self.heuristic(goalState)
                sucessors.append((h, sucessorNode))

        # to the right
        if self.x < self.size - 1:
            if self.matrix[self.x + 1][self.y] != "X":
                sucessorNode = State(self.size, self.matrix, self.x + 1, self.y)
                h = self.heuristic(goalState)
                sucessors.append((h, sucessorNode))

        # upwards
        if self.y > 0:
            if self.matrix[self.x][self.y - 1] != "X":
                sucessorNode = State(self.size, self.matrix, self.x, self.y - 1)
                h = self.heuristic(goalState)
                sucessors.append((h, sucessorNode))

        # to the right
        if self.y < self.size - 1:
            if self.matrix[self.x][self.y + 1] != "X":
                sucessorNode = State(self.size, self.matrix, self.x, self.y + 1)
                h = self.heuristic(goalState)
                sucessors.append((h, sucessorNode))

        # sort the array based on the heuristic
        return sorted(sucessors, key=lambda x: x[0])

    def heuristic(self, goalState: "State"):
        """Uses the Manhattan distance as the heuristic
        @param goalState: State -> the goal state the alg is looking for

        @returns an integer as the heuristic
        """
        return abs(goalState.x - self.x) + abs(goalState.y - self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
