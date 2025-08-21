import torch

# determine reproducibility
seed = torch.Generator().manual_seed(42)
