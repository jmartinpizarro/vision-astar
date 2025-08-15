# vision-astar
This projects aims to develop a neural network able to interpret grid's images (up to 6x6) with an origin coordinate,
a goal coordinate and different obstacles. The neural network will return a matrix that can be simplified into a 
states problem that can be solved with simples search algorithm (based or non based in heuristics).

The project has three parts:

- Dataset: used for training the neural network. A collection of images has been generated. The dataset was created from scratch and you can find in Kaggle. The source code is in the folder `src/synthetic_dataset`.
- Neural Network
- Search Algorithm: implement several of them in order to check performance depending on the algorithm.

Because of the structure of this project, it aims to have a pipeline such as: *Preprocessing* -> *Architecture* -> *Training* -> *Validation*

# Dataset #

The dataset has been created from scratch, using a pen and different types of papers. Images are in constant resolution, using an Iphone SE 2020 for capturing the images.

The grid can have a variable size, obstacles represented as `X`, an origin point represented as `O` and a goal point represented as `G`. 

You can go directly to the dataset using the following [link](https://www.kaggle.com/datasets/javiermartnpizarro/gridpathnet) or you can execute the following command (if you have already setted up your Kaggle API keys, [see this for more info](https://www.kaggle.com/docs/api#authentication) to download it directly using `cURL`):

```bash
mkdir -p data
cd data

curl -L -o gridpathnet.zip\
  https://www.kaggle.com/api/v1/datasets/download/javiermartnpizarro/gridpathnet

unzip gridpathnet.zip

rm gridpathnet.zip
```

If you can generate your own dataset (as the code is non-deterministic), you can run the following command:

```bash
python3 -m scripts.run_dataset_generation
```

# Neural Networks - CNNs #

Different approaches will be considered in order to study the efficiency and size of the different neural networks proposed:

1. **CellNet** – Cell-wise CNN: The input image is split into individual cells, and each one is classified independently (multiclass classification).  
   **Limitation**: This CNN does not infer the grid size; it must be fixed or inferred externally.

2. **GridNet** – CNN with classification grating: A CNN processes the entire image and outputs a `(N, N, 4)` tensor, where each cell is classified into one of the 4 possible symbols (empty, O, G, X). Softmax is applied per cell.  
   Efficient and interpretable.

3. **ConvHeadNet** – CNN with final convolutional head: Similar to GridNet, but instead of flattening, a final `Conv2D` layer with 4 filters is used. Keeps spatial structure and allows efficient per-cell classification via logits.

4. **ViTGrid** – Vision Transformer with patch embedding: A transformer model with patch-wise embedding processes the image as a sequence. It can handle variable-sized grids but is computationally heavier and overkill for this task.

---

## 1. CellNet – Cell-wise CNN

**Description**:  
Splits the full grid image into individual cells (e.g., 3x3 to 10x10 sub-images) and classifies each as one of the 4 possible symbols: `"O"`, `"G"`, `"X"`, or `""`.

**Architecture** (per cell):  
- `Input`: (cell_img, e.g. 100×100×3)  
- `Conv2D(32, 3x3) + ReLU`  
- `MaxPool2D(2x2)`  
- `Conv2D(64, 3x3) + ReLU`  
- `Flatten`  
- `Dense(128) + ReLU`  
- `Dense(4) + Softmax`

**Pros**: Simple, interpretable.  
**Cons**: Requires prior cell segmentation; can't infer grid size.

---

## 2. GridNet – Full CNN with classification grating

**Description**:  
Processes the entire image (e.g., 300×300 px) and outputs a 3D tensor (N×N×4), where N is the grid size. Each spatial position corresponds to a symbol class prediction.

**Architecture**:  
- `Input`: (300×300×3)  
- `Conv2D(32, 3x3, padding='same') + ReLU`  
- `Conv2D(64, 3x3, padding='same') + ReLU`  
- `Conv2D(128, 3x3, padding='same') + ReLU`  
- `Conv2DTranspose(64, 3x3)` (optional upsampling)  
- `Conv2D(4, 1x1)` → logits per cell  
- `Reshape` to (N×N×4), with softmax per cell

**Pros**: End-to-end, infers grid and content.  
**Cons**: Needs grid-size consistency or dynamic handling.

---

## 3. ConvHeadNet – CNN with final convolutional head

**Description**:  
Similar to GridNet but uses a final convolutional layer to produce class logits directly, maintaining spatial structure without flattening.

**Architecture**:  
- `Input`: (300×300×3)  
- `Backbone`: ResNet-like (or MobileNet for lightweight)  
- `Conv2D(4, 1x1)` → output shape: (N×N×4)  
- `Softmax` per (i, j) cell

**Pros**: More efficient, fewer parameters.  
**Cons**: Requires fixed grid size or resizing.

---

## 4. ViTGrid – Vision Transformer (ViT) with patch embedding

**Description**:  
Applies patch-wise tokenization and transformer blocks. Each patch corresponds to a cell, and the model classifies it using global context.

**Architecture**:  
- `Input`: (300×300×3)  
- `Patch Embedding`: (e.g., 10x10 patches)  
- `Transformer Encoder` × L layers  
- `Classification Head` per patch  
- Output: (N×N×4) with per-cell predictions

**Pros**: Flexible with grid size, uses full image context.  
**Cons**: Heavier, needs more data to train effectively.

---

## Summary

| Model        | Grid Size Handling | Context Aware | Lightweight | Notes                         |
|--------------|--------------------|----------------|--------------|-------------------------------|
| CellNet      | ❌ (manual split)   | ❌              | ✅           | Easy but inflexible           |
| GridNet      | ⚠️ (fixed or padded) | ✅              | ✅           | Strong baseline                |
| ConvHeadNet  | ✅ (structured)      | ✅              | ✅✅         | Best efficiency-accuracy tradeoff |
| ViTGrid      | ✅ (dynamic)         | ✅✅             | ❌           | Best generalization, costly   |

# Preprocessing #

There are two types of preprocessing: one for the `CellNet` network and another script for the others networks.

You may execute the scripts for preprocessing with the following command from the source directory:

```bash
python -m scripts.run_preprocessing --input something --output other --network cellnet

```

# Linter and Testing #

For linter, this project uses `pre-commit` and `ruff` (dependencies can be installed in `requirements.txt`). 
If you want to install and execute it in your own environment, do the following steps:

```bash
pre-commit install
pre-commit run --all-files
```
