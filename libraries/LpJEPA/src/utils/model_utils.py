import json
from logging import getLogger
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import PROJECT_NAME
from helper_functions.helpers import get_gpu_usage, load_config, move_batch_to_device
from rt_jepa.model.lpjepa import LpJEPA
from rt_jepa.utils.dataset import CBCTDataset, collate_cbct

logger = getLogger(f'{PROJECT_NAME}.{__name__}')


def make_jepa(
    config_path: Path, patch_size: int, device: torch.device, weights_path: Path | None = None
):

    config = load_config(config_path)

    rt_jepa = LpJEPA(
        # Encoder part
        image_size=config.get('encoder')['image_size'],
        d_model=config.get('encoder')['d_model'],
        patch_size=patch_size,
        num_heads=config.get('encoder')['num_heads'],
        num_encoder_layers=config.get('encoder')['num_layers'],
        mlp_ratio=int(config.get('encoder')['mlp_ratio']),
        mlp_drop=config.get('encoder')['mlp_drop'],
        drop_path_rate=config.get('encoder')['drop_path_rate'],
        num_views=config.get('encoder')['num_views'],
        # Projector part
        proj_dim=config.get('projector')['d_proj'],
        mu=config.get('rdm_reg')['mu'],
        sigma=config.get('rdm_reg')['sigma'],
        p=config.get('rdm_reg')['p'],
        num_projs=config.get('rdm_reg')['num_projs'],
        # Loss part
        lambda_reg=config.get('loss')['lambda_reg'],
        rdm=config.get('rdm_reg')['rdm'],
        K=config.get('rdm_reg')['K'],
    ).to(device)

    if weights_path is not None:
        logger.info(f'Loading model weights from {weights_path}.')
        rt_jepa.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

    return rt_jepa


def embed_databank(
    config_path: Path,
    jepa_weights_path: Path,
    batch_size: int,
    output_dir: Path,
    data_type: str = 'full',
    sample_prop: float = 1.0,
    prop_class: float = 0.5,
    data_source: str = 'all',
):
    logger.info('Embedding data using RT-JEPA.')
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_dir.mkdir(parents=True, exist_ok=True)

    dat: CBCTDataset = CBCTDataset(
        tensors_dir=config.get('data').get('data_out'),
        input_size=config.get('encoder').get('image_size'),
        data_type=data_type,
        sample_prop=sample_prop,
        prop_class=prop_class,
        data_source=data_source,
    )

    dataloader = DataLoader(
        dat,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
        pin_memory=False,
        collate_fn=collate_cbct,
    )

    rt_jepa = torch.load(jepa_weights_path, map_location=device, weights_only=False)

    rt_jepa.eval()

    progress_bar = tqdm(dataloader, leave=True, ncols=50, dynamic_ncols=True)
    all_embeddings = []
    all_paths = []
    for b, batch in enumerate(progress_bar):
        tensor, loc, is_ct, _ = move_batch_to_device(batch, device)
        _, _, _, batch_paths = batch
        batch_emb = rt_jepa.embed_tokens(tensor)
        all_embeddings.append(batch_emb.squeeze(1).detach().cpu())
        all_paths.append([str(path) for path in batch_paths])
        progress_bar.set_postfix(get_gpu_usage())
    embed_bank = torch.vstack(all_embeddings)
    with open(output_dir / 'tensor_paths.json', 'w') as f:
        json.dump(all_paths, f, indent=4)
    logger.info(f'Saving input tensors to {output_dir}/input_tensors.pt')
    torch.save(embed_bank, output_dir / 'embeddings.pt')
