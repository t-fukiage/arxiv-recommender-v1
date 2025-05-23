import logging
import os
import pathlib
import yaml

def load_config(path: pathlib.Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def setup_logger(level: str = os.environ.get("LOGLEVEL", "INFO")):
    logging.basicConfig(
        format="[%(levelname)s %(asctime)s] %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S"
    ) 