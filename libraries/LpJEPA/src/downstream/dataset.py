import torch
import json

from itertools import chain
from torch.utils.data import Dataset
from pathlib import Path


class EmbeddingDataset(Dataset):
    def __init__(self, input_path: Path = Path('data')):
        self.input_path = input_path
        self.embeddings = torch.load(input_path / 'embeddings.pt')
        with open(input_path / 'tensor_paths.json', 'r') as f:
            self.tensor_paths = list(chain.from_iterable(json.load(f)))
        self.d_proj = self.embeddings.shape[1]

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx, :], torch.load(Path(self.tensor_paths[idx]))
