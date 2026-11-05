import json
import logging
from itertools import compress
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pydicom import dcmread
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from rt_jepa.preprocess import dataprep as dp

MAIN_NAME = Path(__file__).parent.parent.parts[-1]
logger = logging.getLogger(f'{MAIN_NAME}.{__name__}')


def random_pick(path_list: list, source_list: list, prop: float):
    rand_filter: list = np.random.choice(2, len(path_list), p=[1 - prop, prop])
    return list(compress(path_list, rand_filter)), list(compress(source_list, rand_filter))


def get_source(path):
    return path._str.split('\\')[-1].split('__')[0]


def get_sourced_path(path_list, source_list, data_source):
    source_filter: list = np.array(source_list) == data_source
    return list(compress(path_list, source_filter)), list(compress(source_list, source_filter))


class CBCTDataset(Dataset):
    def __init__(
        self,
        tensors_dir: str | Path,
        input_size: tuple | int = (512, 512),
        data_type: str = 'full',
        sample_prop: float = 0.3,
        prop_class: float = 0.5,
        data_source: str = 'all',
    ):
        if isinstance(input_size, int):
            input_size = (input_size,)

        self.input_size = input_size
        self.tensors_dir: str | Path = tensors_dir
        self.prop_class = prop_class
        self.data_source = data_source
        self.data_type: str = data_type
        self.sample_prop: float = sample_prop
        self.sources_list = None

        if self.data_type == 'full':
            self.sample_prop: float = 1.0
        ct_path: list = list(Path(self.tensors_dir, 'CT').iterdir())
        cbct_path: list = list(Path(self.tensors_dir, 'CBCT').iterdir())

        ct_sources: list = [get_source(i) for i in ct_path]
        cbct_sources: list = [get_source(i) for i in cbct_path]

        # selection des donnees d'une source specifique
        if self.data_source != 'all':
            ct_path, ct_sources = get_sourced_path(ct_path, ct_sources, self.data_source)
            cbct_path, cbct_sources = get_sourced_path(cbct_path, cbct_sources, self.data_source)

        # Selection aleatoire de ct/cbct dans les bonnes proportions (si "full" alors sample_prop = 1)
        ct_path, ct_sources = random_pick(
            ct_path, ct_sources, prop=self.sample_prop * self.prop_class
        )
        cbct_path, cbct_sources = random_pick(
            cbct_path, cbct_sources, prop=self.sample_prop * (1 - self.prop_class)
        )

        # regroupement des paths, sources et creation des classes
        self.tensors_paths: list = ct_path + cbct_path
        self.tensors_sources: list = ct_sources + cbct_sources
        self.tensors_class: list = np.concatenate(
            [np.ones(len(ct_path), dtype=np.int_), np.zeros(len(cbct_path), dtype=np.int_)]
        ).tolist()
        self.sources_list = list(set(self.tensors_sources))

    def save_tensors_path(self, save_path: Path):
        with open(save_path, 'w') as f:
            json.dump([str(path) for path in self.tensors_paths], f)

    def __len__(self):
        return len(self.tensors_paths)

    def __getitem__(self, item):
        matrix = torch.load(self.tensors_paths[item])  # tensor [512, 512]
        s = self.sources_list.index(self.tensors_sources[item])
        c = self.tensors_class[item]
        return (
            matrix,
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(c, dtype=torch.float32),
            self.tensors_paths[item],
        )

    def plot_img(self, item: int):
        dicom: Path = self.img_paths[item]
        img = dcmread(dicom)
        pixels = (dp.get_pixels(img) * 255).astype(np.uint8)
        pixels = Image.fromarray(pixels, mode='L')
        plt.imshow(pixels)
        plt.axis('off')
        plt.title(f'Image {item}')
        plt.show()


def collate_cbct(batch):
    tensor_data = [sample[:-1] for sample in batch]
    tensor_paths = [sample[-1] for sample in batch]
    collated_tensors = default_collate(tensor_data)
    return *collated_tensors, tensor_paths
