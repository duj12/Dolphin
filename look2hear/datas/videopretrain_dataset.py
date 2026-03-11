import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .transform import get_preprocessing_pipelines


class VideoPretrainDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "",
        segment: float = 4.0,
        is_train: bool = True,
    ):
        super().__init__()
        if data_dir is None:
            raise ValueError("DATA DIR is None!")

        self.data_dir = data_dir
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()[
            "train" if is_train else "val"
        ]
        print("lipreading_preprocessing_func: ", self.lipreading_preprocessing_func)

        if segment is None:
            self.fps_len = None
        else:
            self.fps_len = int(segment * 25)

        self.mouths = []
        with open(self.data_dir, "r", encoding="utf-8") as f:
            for file in f.readlines():
                self.mouths.append(file.strip())
        self.length = len(self.mouths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        source_mouth = self.lipreading_preprocessing_func(np.load(self.mouths[idx])["data"])

        return (
            torch.tensor(np.expand_dims(source_mouth, axis=0), dtype=torch.float32),
            self.mouths[idx],
        )


class VideoPretrainDataModule(object):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        segment: float = 4.0,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        if train_dir is None or valid_dir is None:
            raise ValueError("DATA DIR is None!")

        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.segment = segment
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self) -> None:
        self.data_train = VideoPretrainDataset(
            data_dir=self.train_dir,
            segment=self.segment,
            is_train=True,
        )
        self.data_val = VideoPretrainDataset(
            data_dir=self.valid_dir,
            segment=self.segment,
            is_train=False,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val
