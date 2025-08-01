# vision-astar
This projects aims to develop a neural network able to interpret grid's images (up to 6x6) with an origin coordinate,
a goal coordinate and different obstacles. The neural network will return a matrix that can be simplified into a 
states problem that can be solved with simples search algorithm (based or non based in heuristics).

The project has three parts:

- Dataset: used for training the neural network. A collection of images has been drawn (mainly grids with dimensions between 3-x3 and 6x6). The dataset was created from scratch and you can find in Kaggle.
- Neural Network (don't know what I will use yet)
- Search Algorithm: implement several of them in order to check performance depending on the algorithm.

Because of the structure of this project, it aims to have a pipeline such as: *Preprocessing* -> *Architecture* -> *Training* -> *Validation*

# Dataset #

The dataset has been created from scratch, using a pen and different types of papers. Images are in constant resolution, using an Iphone SE 2020 for capturing the images.

The grid can have a variable size, obstacles represented as `X`, an origin point represented as `F` and a goal point represented as `G`. 

You can go directly to the dataset using the following [link](https://www.kaggle.com/datasets/javiermartnpizarro/handdrawngrid2matrix/data) or you can execute the following command (if you have already setted up your Kaggle API keys, [see this for more info](https://www.kaggle.com/docs/api#authentication) to download it directly using `cURL`).

```bash
mkdir -p data
cd data

curl -L -o handdrawngrid2matrix.zip \
  https://www.kaggle.com/api/v1/datasets/download/javiermartnpizarro/handdrawngrid2matrix

unzip handdrawngrid2matrix.zip

rm handdrawngrid2matrix.zip
```


# Linter and Testing #

For linter, this project uses `pre-commit` and `ruff` (dependencies can be installed in `requirements.txt`). 
If you want to install and execute it in your own environment, do the following steps:

```bash
pre-commit install
pre-commit run --all-files
```
