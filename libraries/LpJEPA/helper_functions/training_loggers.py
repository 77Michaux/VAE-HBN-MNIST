import logging
import csv
import pandas as pd

from math import isnan
from pathlib import Path
from matplotlib import pyplot as plt

logger = logging.getLogger(f"tiny_jepa.{__name__}")


class GradientLogger:
    """
    Get the L2 gradient norm of the model's parameters and log it
    """
    def __init__(self, model):
        self.model = model
        self.total_grad_norm = 0.

    def log_gradients(self):
        total_norm = 0.
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        self.total_grad_norm = total_norm ** .5
        if isnan(self.total_grad_norm):
            logger.error("gradient exploded and is NaN")
        elif self.total_grad_norm > 100:
            logger.warning(f"gradient getting high: {self.total_grad_norm}")
        elif self.total_grad_norm < 1e-6:
            logger.warning(f"gradient getting low: {self.total_grad_norm}")

        return self.total_grad_norm


class AverageMeter:
    def __init__(self):
        self.count = None
        self.avg = None
        self.val = None
        self.sum = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def log_loss(self, epoch, batch):
        logger.info(f"average loss at epoch {epoch} batch {batch}: {self.avg:,.4f}")


class CsvLogger:
    def __init__(self, file_path: Path, csv_name: str, headers: list):
        self.file_path = file_path
        self.headers = headers
        self.csv_name = csv_name

        Path(file_path).mkdir(parents=True, exist_ok=True)

        with open(Path(self.file_path / self.csv_name), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log(self, data):
        with open(Path(self.file_path / self.csv_name), "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data)


class MetricTracker:
    def __init__(self, *keys):
        # On crée un AverageMeter pour chaque clé fournie au départ
        self._data = {k: AverageMeter() for k in keys}

    def update(self, key, value, n=1):
        # Si la clé n'existe pas encore, on la crée à la volée
        if key not in self._data:
            self._data[key] = AverageMeter()
        self._data[key].update(value, n)

    def update_all(self, metrics_dict, n=1):
        """
        Prend un dict {'loss': 0.5, 'acc': 98.2} et update tout d'un coup.
        """
        for key, value in metrics_dict.items():
            if key not in self._data:
                self._data[key] = AverageMeter()

            # Gestion automatique des tenseurs PyTorch pour éviter les fuites
            if hasattr(value, "item"):
                value = value.item()
            self._data[key].update(value, n)

    def reset(self):
        for meter in self._data.values():
            meter.reset()

    def result(self):
        # Retourne un dictionnaire avec toutes les moyennes actuelles
        return {k: v.avg for k, v in self._data.items()}

    def __getattr__(self, name):
        # Permet d'accéder à une métrique via tracker.loss par exemple
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"Metric '{name}' not found.")

    def __str__(self):
        # Un résumé propre pour ton logging
        return " | ".join([f"{k}: {v.avg:.4f}" for k, v in self._data.items()])


def plot_loss(csv_path: str | Path,
              save_dir: str | Path = "logs",
              model_name: str = "NOMOS",
              save_name: str = "loss.pdf"):
    csv_path = Path(csv_path)
    save_dir = Path(save_dir)
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: loss log file not found at {csv_path}")

    if "Epoch" not in df.columns:
        print("Error: CSV file must contain 'Epoch'.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(12, 7))
    for loss in df.columns[1:]:
        plt.plot(df["Epoch"], df[loss], linestyle="-", label=loss, alpha=1, marker=".")
    plt.title(f"{model_name} training over {df.iloc[-1, 0]} epochs", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Average metric", fontsize=12)
    plt.legend()
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(Path(save_dir, f"{model_name}_{save_name}.pdf"))
    logger.info(f"saved pdf at {save_dir }/{model_name}_{save_name}")
    plt.close()
