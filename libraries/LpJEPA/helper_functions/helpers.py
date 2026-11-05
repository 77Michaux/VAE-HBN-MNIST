import json
import torch
import logging
import requests
import tkinter as tk
import os
import numpy as np
import random

from tkinter import filedialog
from pathlib import Path
from torchinfo import summary
from typing import Union, Mapping, Sequence

logger = logging.getLogger(f"diab_ts.{__name__}")

# ---------- GPU Monitoring Import ----------

try:
    import pynvml
    pynvml.nvmlInit()
    pynvml_available = True
except (ImportError, pynvml.NVMLError):
    pynvml_available = False


def load_config(config_path: Path):
    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_data_dir(config: dict):
    if not os.path.exists(config["data"]["data_dir"]):
        root = tk.Tk()
        root.withdraw()
        new_data_dir = filedialog.askdirectory(title="Select data directory")

        config["data"]["data_dir"] = new_data_dir

    return config


def set_devices():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"PyTorch version {torch.__version__}")
        logger.info(f"CUDA version {torch.version.cuda}")
        logger.info(f"found {device_count} CUDA_enabled GPU(s)")
        current_device = torch.cuda.current_device()
        logger.info(f"current default CUDA: index = {current_device}, "
                    f"name = {torch.cuda.get_device_name(current_device)}")
        return torch.device("cuda")

    else:
        logger.info("no CUDA detected, using CPU.")
        return torch.device("cpu")


def move_batch_to_device(batch: Union[torch.Tensor, Sequence, Mapping], device: torch.device):
    """Recursively moves batch elements (tensors) to the specified device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        # Handle lists/tuples
        return type(batch)(move_batch_to_device(item, device) for item in batch)
    elif isinstance(batch, dict):
        # Handle dictionaries (like your **batch)
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    else:
        # Return non-tensor data as is
        return batch



def get_gpu_usage(device_id=0):
    """Returns GPU utilization and memory usage in a dictionary."""
    if not pynvml_available:
        return {}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_util": f"{utilization.gpu}%",
            "mem_used": f"{memory.used / memory.total * 100:.0f}%"
            # 'mem_used': f"{memory.used // 1024**2}MB",
            # 'mem_total': f"{memory.total // 1024**2}MB"
        }
    except pynvml.NVMLError:
        return {}


def save_checkpoint(model, optimizer, scheduler, file_path):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }
    torch.save(state, file_path)


def send_notification(config, final_loss, epochs):
    """Sends a push notification using ntfy.sh when training is complete."""
    notif_config = config.get("notifications", {})
    if not notif_config.get("enabled"):
        return
    topic = notif_config.get("ntfy_topic")
    if not topic:
        logger.warning("Notifications enabled, but 'ntfy_topic' is not set.")
        return
    try:
        data_payload = (f"Training complete after {epochs} epochs.\n\n"
                        f"Final average loss: {final_loss:.4f}").encode("utf-8")

        response = requests.post(
            f"https://ntfy.sh/{topic}",
            data=data_payload,
            headers={
                "Title": "I-JEPA Training Complete!",
                "Priority": "default",
                "Tags": "tada",
                "Markdown": "true",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
        response.raise_for_status()
        logger.info(f"Completion notification sent to ntfy topic: {topic}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send notification due to network error: {e}")
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")


def summarize_model(model,
                    batch_size,
                    patch_size,
                    config,
                    device,
                    depth: int = 4,
                    summary_filename: str = "logs/model_summary.txt"):
    image_size = config["encoder"]["image_size"]
    num_patches = (image_size // patch_size) ** 2

    images_sample = torch.randn(batch_size, 1, image_size, image_size).to(device)
    class_sample = torch.zeros(batch_size, dtype=torch.float32).to(device)

    print(f"\n--- RT-JEPA Model Summary: input size = {image_size}, num patches = {num_patches} ---\n")
    model_summary = summary(
        model=model,
        input_data=[images_sample, class_sample],
        depth=depth,
        verbose=0
    )

    summary_str = str(model_summary)

    with open(Path(summary_filename), "w", encoding="utf-8") as f:
        f.write(summary_str)
    print(f"Model summary saved to {summary_filename}")
