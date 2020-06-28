# Core Library
import logging

# Third party
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(id="CityEnv-v0", entry_point="gym_cityenv.envs:CityEnv")
