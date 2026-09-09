from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pytorch_lightning.callbacks import Callback


class TrainingHistoryCallback(Callback):
    def __init__(self, run_dir):
        super().__init__()

        self.run_dir = Path(run_dir)
        self.history_dir = self.run_dir / "training_history"
        self.graph_dir = self.run_dir / "graphs"

        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.graph_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.history_dir / "training_history.csv"

        # epoch -> {"train_loss": ..., "val_loss": ...}
        self.history = {}

        # Load existing history when resuming
        if self.csv_path.exists():
            with open(self.csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    if not row.get("epoch"):
                        continue

                    epoch = int(row["epoch"])
                    train_loss = (
                        float(row["train_loss"])
                        if row.get("train_loss")
                        else None
                    )
                    val_loss = (
                        float(row["val_loss"])
                        if row.get("val_loss")
                        else None
                    )

                    self.history[epoch] = {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    }

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        train_loss = metrics.get("train_loss")

        if train_loss is None:
            return

        epoch = trainer.current_epoch + 1
        train_loss = float(train_loss.detach().cpu())

        val_loss_value = None

        check_every = trainer.check_val_every_n_epoch

        if check_every is not None and epoch % check_every == 0:
            val_loss = metrics.get("val_loss")

            if val_loss is not None:
                val_loss_value = float(val_loss.detach().cpu())

        # Replace existing epoch if it is repeated after resume
        self.history[epoch] = {
            "train_loss": train_loss,
            "val_loss": val_loss_value,
        }

        self._save_csv()
        self._save_graph()

    def _save_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

            for epoch in sorted(self.history):
                values = self.history[epoch]

                writer.writerow([
                    epoch,
                    values["train_loss"]
                    if values["train_loss"] is not None
                    else "",
                    values["val_loss"]
                    if values["val_loss"] is not None
                    else "",
                ])

    def _save_graph(self):
        epochs = sorted(self.history)

        train_epochs = []
        train_losses = []

        val_epochs = []
        val_losses = []

        for epoch in epochs:
            values = self.history[epoch]

            if values["train_loss"] is not None:
                train_epochs.append(epoch)
                train_losses.append(values["train_loss"])

            if values["val_loss"] is not None:
                val_epochs.append(epoch)
                val_losses.append(values["val_loss"])

        plt.figure(figsize=(8, 5))

        if train_losses:
            plt.plot(
                train_epochs,
                train_losses,
                marker="o",
                label="Train loss",
            )

        if val_losses:
            plt.plot(
                val_epochs,
                val_losses,
                marker="o",
                label="Validation loss",
            )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("TFM-SDE Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        output_file = self.graph_dir / "train_validation_loss.png"

        plt.savefig(output_file, dpi=300)
        plt.close()