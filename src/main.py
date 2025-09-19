from algorithm.ida_astar import ida_astar
from algorithm.State import State


def main():
    matrix = [["", ""], ["", ""]]

    init = State(2, matrix, 0, 0)
    goal = State(2, matrix, 1, 1)

    path = ida_astar(init, goal)

    if path == -1:
        print("Seems that no possible solution could be calculated\n")
        return 0

    print("A solution was calculated for the problem:\n")

    for state in path[0]:
        print(state)
        print(f"({state.x},{state.y}) -> ")

    print(
        f"A total of {len(path) - 2} movements were need to achieved the final solution\n"
    )

    return 0


if __name__ == "__main__":
    main()
