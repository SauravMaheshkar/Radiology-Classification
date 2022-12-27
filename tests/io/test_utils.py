"""Test utility functions."""
import pytest

from src.io.utils import preprocess_chexpert


@pytest.mark.chexpert
@pytest.mark.parametrize("policy", [("ones"), ("zeroes"), ("random")])
def test_preprocess_chexpert(policy: str) -> None:
    """Test the preprocess_chexpert function."""

    prefix: str = "tests/data/chexpert/"

    processed_dataframe = preprocess_chexpert(
        filepath="tests/data/chexpert/train.csv",
        prefix=prefix,
        policy=policy,
    )

    assert len(processed_dataframe) == 4
    assert processed_dataframe.columns.tolist() == ["Path", "Label"]
    assert prefix in processed_dataframe["Path"].values[0]
