from typing import Dict, Any
import json
from pathlib import Path


def save_params(param_dict: Dict[str, Any], base_folder: str | Path):
    base_folder = Path(base_folder)
    with open(base_folder.joinpath("params.json"), "w") as file:
        json.dump(param_dict, file, indent=4)
