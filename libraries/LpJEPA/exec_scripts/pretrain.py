import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from helper_functions.helpers import (get_gpu_usage, load_config, save_checkpoint, set_seed, move_batch_to_device,
                                      summarize_model)
from helper_functions.schedulers import WarmupCosineSchedule
from helper_functions.training_loggers import MetricTracker, CsvLogger, GradientLogger
from src.dataset import CBCTDataset, collate_cbct
from src.model.lpjepa import LpJEPA


try:
    import pynvml

    pynvml.nvmlInit()
    pynvml_available = True
except (ImportError, pynvml.NVMLError):
    pynvml_available = False


def score_model(metrics: dict, lambda_inv: float):
    # Stability (lower = better)
    inv = metrics['inv']
    classi = metrics['class'] if 'class' in metrics else 0
    # Gap to RGG (lower = better)
    rdm = metrics['rdm']
    denominator = lambda_inv * (0.8 * np.log1p(inv) + 0.2 * np.log1p(classi)) + (1 - lambda_inv) * np.log1p(rdm) + 1e-9
    return 1.0 / denominator


def pretrain(
        data_type: str,
        epochs: int = 1,
        batch_size: int = 16,
        patch_size: int = 16,
        num_workers: int = 1,
        use_class: bool = True
):

    config = load_config(Path("config", "config.json"))

    set_seed(config.get("seed"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dat: CBCTDataset = CBCTDataset(
        tensors_dir=config.get("data").get("data_out"),
        input_size=config.get("encoder").get("image_size"),
        data_type="sample",
        sample_prop=.1,
        prop_class=.5,
        data_source="all",
    )

    print(f"{dat.sources_list=}")

    dataloader = DataLoader(dat, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
                            persistent_workers=True, pin_memory=True, collate_fn=collate_cbct)

    model_dir = Path("model")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "rt_jepa.pt"

    rt_jepa = LpJEPA(
        # Encoder part
        image_size=config.get("encoder")["image_size"],
        d_model=config.get("encoder")["d_model"],
        patch_size=patch_size,
        num_heads=config.get("encoder")["num_heads"],
        num_encoder_layers=config.get("encoder")["num_layers"],
        mlp_ratio=int(config.get("encoder")["mlp_ratio"]),
        mlp_drop=config.get("encoder")["mlp_drop"],
        drop_path_rate=config.get("encoder")["drop_path_rate"],
        num_views=config.get("encoder")["num_views"],
        # Projector part
        proj_dim=config.get("projector")["d_proj"],
        mu=config.get("rdm_reg")["mu"],
        sigma=config.get("rdm_reg")["sigma"],
        p=config.get("rdm_reg")["p"],
        num_projs=config.get("rdm_reg")["num_projs"],
        # Loss part
        lambda_inv=config.get("loss")["lambda_inv"],
        rdm=config.get("rdm_reg")["rdm"],
        knots=config.get("rdm_reg")["knots"]
    ).to(device)

    # summarize_model(model=rt_jepa, batch_size=batch_size, patch_size=patch_size, config=config, device=device, depth=4)

    optimizer=torch.optim.AdamW(params=rt_jepa.parameters(), lr=config["training"]["start_lr"])
    warmup_steps = len(dataloader) * int(epochs * config["training"]["warmup_ratio"])
    total_steps = len(dataloader) * epochs
    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        T_max=total_steps,
        start_lr=config["training"].get("start_lr"),
        ref_lr=config["training"].get("ref_lr"),
        final_lr=config["training"].get("final_lr")
    )

    metrics_list = ["loss", "inv", "rdm", "sparse_prop"]
    if use_class:
        metrics_list.insert(3, "class")
    metrics_tracker = MetricTracker(*metrics_list)

    grad_logger = GradientLogger(rt_jepa)

    csv_headers = ["Epoch", "Loss", "MSE", "RDMReg", "Sparsity"]
    if use_class:
        csv_headers.insert(4, "Class")
    csv_loss_logger = CsvLogger(
        file_path=Path("logs"),
        csv_name="losses.csv",
        headers=["Epoch", "Loss", "MSE", "RDMReg", "Sparsity"]
    )
    csv_grad_logger = CsvLogger(
        file_path=Path("logs"),
        csv_name="gradients.csv",
        headers=["Epoch", "Batch", "gradient_norm"]
    )

    checkpoint_dir = Path("checkpoints")
    checkpoint_freq = config["logging"].get("checkpoint_freq")

    best_model_score = -np.inf

    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, leave=True, ncols=50, dynamic_ncols=True, desc=f"Epoch {epoch+1}/{epochs}")
        for b, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            # data, _ = batch
            tensor, loc, is_ct, _ = move_batch_to_device(batch, device)
            if not use_class:
                is_ct = None
            results = rt_jepa(tensor, is_ct)

            metrics_dict = dict(zip(metrics_list, results))
            metrics_tracker.update_all(metrics_dict, batch_size)
            results[0].backward()
            grad_norm = grad_logger.log_gradients()
            csv_grad_logger.log([epoch+1, b+1, grad_norm])

            optimizer.step()
            scheduler.step()

            postfix_stats = metrics_tracker.result()
            postfix_stats["lr"] = optimizer.param_groups[0]["lr"]
            postfix_stats["grad"] = grad_norm
            if device.type == "cuda":
                postfix_stats.update(get_gpu_usage())
            progress_bar.set_postfix(postfix_stats)

        if (epoch+1) % checkpoint_freq == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{epoch+1}.pth"
            save_checkpoint(rt_jepa, optimizer, scheduler, checkpoint_path)

        csv_loss_logger.log([epoch+1, *metrics_tracker.result().values()])

        model_quality_score = score_model(metrics_tracker.result(), lambda_inv=config.get("loss")["lambda_inv"])

        if model_quality_score > best_model_score:
            best_model_score = model_quality_score
            print(f'Current best model score: {best_model_score:.4f}')
            save_path = Path(model_path)
            torch.save(rt_jepa, save_path)
