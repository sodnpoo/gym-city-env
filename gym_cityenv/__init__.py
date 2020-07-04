# Core Library
import logging

# Third party
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(id="CityEnv-v0", entry_point="gym_cityenv.envs:CityEnv")
register(id="CityEnvForest-v0", entry_point="gym_cityenv.envs:CityEnvForest")
register(id="CityEnvRewardByPower-v0", entry_point="gym_cityenv.envs:CityEnvRewardByPower")
register(id="CityEnvEx1-v0", entry_point="gym_cityenv.envs:CityEnvEx1")
