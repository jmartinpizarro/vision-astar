from src.algorithm.State import State

import math
from typing import List


def ida_astar(root: State, goal: State):
    """
    IDA* Implementation

    @param root: State -> the initial state of the problem
    @param goal: State -> the goal state of the problem
    """

    threshold = root.heuristic(goal)
    path = [root]

    while 1:
        # g = the real cost of arriving to the current node = 0
        t = search(goal, path, 0, threshold)

        if t == "FOUND":
            return (path, threshold)

        if t == math.inf:
            return -1

        threshold = t

    return path


def search(goal: State, path: List["State"], g: int, threshold: int):
    """
    search backtracking algorithm used in IDA

    @param goal: State -> the goal state of the problem
    @param path: List["State"] -> current path
    @param g: int -> the real cost of arriving to the current node
    @param threshold: int -> the bound of the algorithm.

    @returns "FOUND" if the goal node has been founded, inf otherwise
    """

    node = path[-1]  # take the last node of the path (root in the first iter)
    f = g + node.heuristic(goal)

    if f > threshold:
        return f

    if node == goal:
        return "FOUND"

    min_threshold = math.inf

    for children in node.children(goal):
        child_node = children[1]

        if child_node not in path:
            path.append(child_node)
            t = search(goal, path, g + node.heuristic(child_node), threshold)

            if t == "FOUND":
                return "FOUND"

            if t < min_threshold:
                min_threshold = t

            path.pop()

    return min_threshold
