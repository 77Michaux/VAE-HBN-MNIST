import torch

from pytorch_msssim import ssim
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from helper_functions.helpers import (get_gpu_usage, load_config, save_checkpoint, set_seed, move_batch_to_device)
from src.downstream.dataset import EmbeddingDataset
from src.add_on.decoder import ImageConvDecoder
from helper_functions.schedulers import WarmupCosineSchedule


def train_decoder(config_path: Path,
                  batch_size: int,
                  num_workers: int,
                  epochs: int = 100):

    config_dec = load_config(config_path)

    set_seed(config_dec.get("seed"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EmbeddingDataset(Path('data'))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                            persistent_workers=True)

    dec_weights_dir = Path("downstream")
    dec_weights_dir.mkdir(parents=True, exist_ok=True)
    dec_weights_path = dec_weights_dir / "ds_decoder.pt"

    decoder = ImageConvDecoder(d_in=dataset.d_proj,
                               linear_depth=None).to(device)

    optimizer = torch.optim.AdamW(params=decoder.parameters(), lr=config_dec["training"]["start_lr"])
    warmup_steps = len(dataloader) * int(epochs * config_dec["training"]["warmup_ratio"])
    total_steps = len(dataloader) * epochs
    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        T_max=total_steps,
        start_lr=config_dec["training"].get("start_lr"),
        ref_lr=config_dec["training"].get("ref_lr"),
        final_lr=config_dec["training"].get("final_lr")
    )

    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        for b, batch in enumerate(progress_bar):
            emb, img = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            img_pred = decoder(emb)
            loss = torch.tensor(1) - ssim(img, img_pred, data_range=1.0, size_average=True, nonnegative_ssim=True)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
            scheduler.step()


if __name__ == "__main__":
    config_dir: Path = Path("config")
    batch_size: int = 64
    num_workers: int = 1
    patch_size: int = 32
    jepa_weights_path: Path = Path("model", "rt_jepa.pt")
    databank_dir = Path("data")
    train_decoder(config_path=config_dir / 'config_decoder.json',
                  batch_size=batch_size,
                  num_workers=num_workers,
                  epochs=1)
