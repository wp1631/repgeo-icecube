from typing import Tuple
from pathlib import Path


def ensure_get_data_folder(
    basepath: str | Path, data_folder_name: str = "data"
) -> Tuple[Path, Path]:
    basepath = Path(basepath)
    rawdat_path = basepath.joinpath(data_folder_name, "rawdat")
    img_path = basepath.joinpath(data_folder_name, "img")
    if not rawdat_path.exists():
        rawdat_path.mkdir(parents=True)
    if not img_path.exists():
        img_path.mkdir(parents=True)
    return (rawdat_path, img_path)
