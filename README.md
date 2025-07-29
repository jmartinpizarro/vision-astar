# vision-astar
This projects aims to develop a neural network able to interpret grid's images (up to 10x10) with an origin coordinate,
a goal coordinate and different obstacles. The neural network will return a matrix that can be simplified into a 
states problem that can be solved with simples search algorithm (based or non based in heuristics).

The project has three parts:

- Dataset: used for training the neural network. A collection of images has been drawn (mainly grids with dimensions between 3-x3 and 6x6). You can find the entire dataset in the `data/` folder with the corresponding (if needed) labels.
- Neural Network (don't know what I will use yet)
- Search Algorithm: implement several of them in order to check performance depending on the algorithm.

# Linter and Testing #

For linter, this project uses `pre-commit` and `ruff` (dependencies can be installed in `requirements.txt`). 
If you want to install and execute it in your own environment, do the following steps:

```bash
pre-commit install
pre-commit run --all-files
```
