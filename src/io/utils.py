"""Common utilities for IO."""

from typing import List

import pandas as pd

from src.io.constants import CHEXPERT_CLASSES


def preprocess_chexpert(
    filepath: str, prefix: str = "/kaggle/input/chexpert/", policy: str = "ones"
) -> pd.DataFrame:
    """
    Preprocess the CheXpert dataset.

    Args:
        filepath (str): Path to the CheXpert dataset csv file.
        prefix (str): Prefix to add to the path column.
        policy (str): Policy to use for the -1 labels. Can be "ones" or "zeroes".

    Returns:
        pd.DataFrame: A dataframe with the paths and the labels.
    """
    # create a dataframe
    dataframe = pd.read_csv(filepath)

    # append the path column relative to the kaggle dataset path
    dataframe["Path"] = dataframe["Path"].apply(lambda x: prefix + x)

    # create a list of all the paths
    paths_collection = list(dataframe["Path"])

    # preprocess the labels (eliminate Nan, convert -1) and create a list
    labels_collection = preprocess_chexpert_labels(dataframe, CHEXPERT_CLASSES, policy)

    # create a new simpler dataframe with just the paths and labels
    processed_dataframe = pd.DataFrame(
        list(zip(paths_collection, labels_collection)), columns=["Path", "Label"]
    )

    return processed_dataframe


def preprocess_chexpert_labels(
    dataframe: pd.DataFrame, columns: List[str], policy: str = "ones"
) -> List[List[float]]:
    """
    Preprocess the labels of the CheXpert dataset.

    Args:
        dataframe (pd.DataFrame): A dataframe with the labels.
        columns (List[str]): A list of the columns that are being considered as labels.
        policy (str): Policy to use for the -1 labels. Can be "ones" or "zeroes".

    Returns:
        List[List[float]]: A list of the labels for every row.
    """
    labels_collection: List[List[float]] = []

    # iterate over the columns with information about classes
    for _, row in dataframe[columns].iterrows():
        for idx, _ in enumerate(row):
            if row[idx]:
                temp_label = float(row[idx])
                if temp_label == 1:
                    row[idx] = 1
                elif temp_label == -1:
                    if policy == "ones":
                        row[idx] = 1
                    elif policy == "zeroes":
                        row[idx] = 0
                    else:
                        row[idx] = 0
                else:
                    row[idx] = 0
            else:
                row[idx] = 0

        labels_collection.append(list(row))

    return labels_collection
