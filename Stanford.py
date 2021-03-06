import os, requests
import tarfile

from PIL import Image

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl


# Getting the dataset
BASE_PATH = "http://vision.stanford.edu/aditya86/ImageNetDogs/"
IMAGES_NAME = "images.tar"
IMAGES_PATH = BASE_PATH + IMAGES_NAME


def download_dataset():
    print("Fetching the Stanford Dogs dataset...")
    # Check if the files doesn't exist already
    if not os.path.exists(IMAGES_NAME):
        print("Downloading images...")
        images = requests.get(IMAGES_PATH, allow_redirects=True)
        open(IMAGES_NAME, "wb").write(images.content)

    # Extracting the files
    tar = tarfile.open(IMAGES_NAME)
    tar.extractall(os.getcwd())
    tar.close()

    os.remove(IMAGES_NAME)
    print("Download successfully done!")


class DogsDataset(Dataset):
    def __init__(self, root_dir="Images/", train_split=0.7, train=True, transform=None):
        self.train = train
        self.train_split = train_split
        self.root_dir = root_dir
        self.transform = transform

        self.imgs = []
        
        for lab_index, label in enumerate(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            labels_paths = os.listdir(label_path)
            
            if self.train:
                labels_paths = labels_paths[int(len(labels_paths) * train_split):]
            else:
                labels_paths = labels_paths[:int(len(labels_paths) * train_split)]

            for image_path in labels_paths:
                self.imgs.append(
                    {"path": os.path.join(label_path, image_path), "label": lab_index}
                )

    def __len__(self):
        if self.train:
            return int(len(self.imgs) * self.train_split)
        else:
            return int(len(self.imgs) * (1 - self.train_split))

    def __getitem__(self, idx):
        label = self.imgs[idx]["label"]
        img_path = self.imgs[idx]["path"]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


class DogsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, train_split=0.8, root_dir="Images/", download=False, num_workers=1):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.train_split = train_split
        self.num_workers = num_workers

        if download:
            download_dataset()

        self.train_transformer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=(224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
        self.test_transformer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=(224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def setup(self, stage=None):
        if stage in (None, "fit"):
            dogs_dataset = DogsDataset(
            root_dir=self.root_dir, train_split=self.train_split, train=True, transform=self.train_transformer
            )
            train_size = int(len(dogs_dataset) * self.train_split)
            val_size = len(dogs_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(dogs_dataset, [train_size, val_size])

        if stage in (None, "test"):
            self.test_dataset = DogsDataset(
            root_dir=self.root_dir, train_split=self.train_split, train=False, transform=self.test_transformer
            )
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
