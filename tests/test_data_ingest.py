from pathlib import Path

from migro.data.ingest import download_dataset


def test_dataset_path():
    path = download_dataset()

    assert isinstance(path, Path)
    assert path.name == "coupjava-coarse.jsonl"