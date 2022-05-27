from torch import optim, nn
from torchmetrics import Accuracy

import pytorch_lightning as pl
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
        self.save_hyperparameters()

        self.model = models.efficientnet_b3(pretrained=True)
        # Adding linear layer with input 1000 and output num_classes
        self.model.classifier = nn.Sequential(
            nn.Dropout2d(0.3), nn.Linear(1536, num_classes), nn.Softmax(dim=1)
        )

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        return opt

    def training_step(self, batch, batch_idx):
        image, label = batch
        res_label = self.model(image)

        loss = self.loss(res_label, label)
        pre = Precision(num_classes=self.num_classes)

        self.log("train_loss", loss)
        self.log("train_precision", pre(res_label, label))

        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        res_label = self.model(image)
        loss = self.loss(res_label, label)
        pre = Precision(num_classes=self.num_classes)

        self.log("val_loss", loss)
        self.log("val_acc", pre(res_label, label))

        return loss


if __name__ == "__main__":

    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument(
        "--root_dir", type=str, default="Images/", help="Root directory"
    )
    parser.add_argument("--train_split", type=float, default=0.8, help="Train split")
    parser.add_argument("--download", type=bool, default=False, help="Download dataset")
    args = parser.parse_args()

    efficientNet = EfficientNet(
        epochs=args.epochs, lr=args.lr, wd=args.wd, num_classes=NUM_CLASSES
    )
    dogsData = DogsDataModule(
        root_dir=args.root_dir, train_split=args.train_split, download=args.download
    )

    trainer = pl.Trainer(max_epochs=10, default_root_dir="./models")
    trainer.fit(efficientNet, dogsData)
