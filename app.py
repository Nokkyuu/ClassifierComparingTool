import logging
from logging.config import fileConfig
from pathlib import Path
import json
import os

os.chdir(Path(__file__).parent)
fileConfig("./logging.ini")

with open("config.json", "r") as f:
    config = json.load(f)

logger = logging.getLogger(config["logger"]["debug"])