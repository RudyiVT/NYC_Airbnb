import yaml
import logging
import importlib

with open("config.yml", "r") as fp:
    config = yaml.safe_load(fp)

def get_model_module(model_name: str, import_module: str, model_params: dict):
    """Import package from string representation."""
    model_class = getattr(importlib.import_module(import_module), model_name)
    model = model_class(**model_params)
    return model


def set_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
    logger: logging.Logger = logging.getLogger()