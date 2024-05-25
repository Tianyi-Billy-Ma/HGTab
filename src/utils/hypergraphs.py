from torch.utils.data.sampler import BatchSampler
from torch_geometric.data import Data
import torch
from torch.utils.data import default_collate
import random
import os
import numpy as np


class BipartiteData(Data):
    # def __getitem__(self, key, *args, **kwargs):
    #     if key == "reversed":
    #         if torch.all(self.reversed):
    #             return True
    #         elif torch.all(~self.reversed):
    #             return False
    #         else:
    #             raise ValueError("reversed is not consistent")
    #     else:
    #         return super().__getitem__(key, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)
