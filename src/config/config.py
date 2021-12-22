
from dataclasses import dataclass
import logging
import yaml

from src.config.directories import directories as dirs
from src.constants import c_DEV


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    model: dict


def load_config_file():

    read_data = []
    try:
        with open(str(dirs.config) + "/" + c_DEV, 'r') as fp:
            read_data.append(yaml.safe_load(fp))
    except:
            logger.info("yml file don't find inside directories")

    return read_data


def context_choice(ENV):

    return None


def get_config() -> Config:
    
    #configs = context_choice(env)
    config = load_config_file()

    return Config(*config)