import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


class TinyRegressor(pl.LightningModule):
    def __init__(self, in_features: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def make_synthetic_data(n: int = 1024, in_features: int = 10):
    g = torch.Generator().manual_seed(42)
    X = torch.randn(n, in_features, generator=g)
    true_w = torch.randn(in_features, 1, generator=g)
    y = X @ true_w + 0.1 * torch.randn(n, 1, generator=g)
    return TensorDataset(X, y)


def main():
    ds = make_synthetic_data()
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

    model = TinyRegressor(in_features=10)

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=dl)

    # Report the device used
    try:
        device = trainer.strategy.root_device
    except Exception:
        device = model.device
    print(f"Lightning smoke complete. Using device: {device}")


if __name__ == "__main__":
    main()
