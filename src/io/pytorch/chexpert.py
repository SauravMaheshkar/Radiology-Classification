"""Custom PyTorch dataset for CheXpert dataset"""
import typing
from typing import Callable, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CheXpertDataSet(Dataset):
    """Custom PyTorch dataset for CheXpert dataset"""

    def __init__(
        self, processed_dataframe: pd.DataFrame, transform: Optional[Callable] = None
    ) -> None:
        """
        Initialize the dataset

        Args:
            processed_dataframe (pd.DataFrame): dataframe with the paths and the labels.
            transform (callable, optional): Optional transform to be applied per sample.
        """
        self.image_paths = list(processed_dataframe["Path"])
        self.labels = list(processed_dataframe["Label"])
        self.transform = transform

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Take the index of item and returns the image and its labels

        Args:
            index (int): Index, used for indexing the image_paths and labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image and its labels.
        """
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)
        return typing.cast(torch.Tensor, image), torch.FloatTensor(label)

    def __len__(self) -> int:
        """
        Returns the length of the dataset
        """
        return len(self.image_paths)
