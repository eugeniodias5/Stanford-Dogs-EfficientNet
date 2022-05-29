from torch import optim, nn, argmax
from torchmetrics.functional import accuracy

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from torchmetrics import Precision
from torchvision import models

from Stanford import DogsDataModule

import argparse

NUM_CLASSES = 107

# Defining EfficientNet on pytorch lightning
class EfficientNet(pl.LightningModule):
    def __init__(self, epochs, lr=1e-3, wd=1e-2, num_classes=NUM_CLASSES) -> None:
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.num_classes = num_classes

        # Ussing cross-entropy loss
        self.loss = nn.CrossEntropyLoss()
        self.pre = Precision(num_classes=self.num_classes)
        self.save_hyperparameters()

        self.model = models.efficientnet_b3(pretrained=True)
        # Adding linear layer with input 1000 and output num_classes
        self.model.classifier = nn.Sequential(
            nn.Dropout2d(0.3), nn.Linear(1536, 50), nn.Linear(50, num_classes)
        )

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        return opt

    def training_step(self, batch, batch_idx):
        image, label = batch
        res_label = self.model(image)

        loss, acc = self.get_loss_acc(label, res_label)

        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        res_label = self.model(image)

        loss, acc = self.get_loss_acc(label, res_label)

        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, prog_bar=True, logger=True)
        return metrics
        
    def test_step(self, batch, batch_idx):
        image, label = batch
        res_label = self.model(image)

        loss, acc = self.get_loss_acc(label, res_label)

        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, prog_bar=True, logger=True)
        return metrics

    def get_loss_acc(self, label, res_label):
        loss = self.loss(res_label, label)
        res_label = nn.Softmax(dim=1)(res_label)
        logits = argmax(res_label, dim=1)
        acc = accuracy(logits, label)

        return loss, acc

if __name__ == "__main__":

    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--root_dir", type=str, default="Images/", help="Root directory"
    )
    parser.add_argument("--train_split", type=float, default=0.7, help="Train split")
    parser.add_argument("--download", type=bool, default=False, help="Download dataset")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers")

    args = parser.parse_args()

    efficient_net = EfficientNet(
        epochs=args.epochs, lr=args.lr, wd=args.wd, num_classes=NUM_CLASSES
    )
    dogs_data = DogsDataModule(
        root_dir=args.root_dir, train_split=args.train_split, download=args.download, num_workers=args.num_workers
    )

    
    logger = CSVLogger("logs", name="model_logs")

    trainer = pl.Trainer(max_epochs = args.epochs, default_root_dir="./models", accelerator="gpu", devices=[1], logger=logger)
    trainer.fit(efficient_net, dogs_data)
    trainer.test(efficient_net, dataloaders=dogs_data)
